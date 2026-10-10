#include "app/claude_bridge.h"

#include "app/claude_tool_catalog.h"
#include "app/search_controller.h"
#include "input_timeline_time.h"
#include "mutations/input_event_formatter.h"
#include "mutations/input_event_utils.h"
#include "replay_file_io.h"
#include "searches/algorithm_registry.h"
#include "searches/search_runner.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLocalServer>
#include <QLocalSocket>
#include <QQmlApplicationEngine>
#include <QQuickWindow>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QStandardPaths>
#include <QTextStream>
#include <QTimer>

#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace forevertas::app {

namespace fv = forevervalidator;
namespace fvx = forevervalidator::experimental;

namespace {

constexpr std::uint32_t kTickMs = kInputTimelineTickDurationMs;
constexpr std::uint32_t kDefaultSimulateHorizonMs = 600000u;
constexpr int kMaximumCompareScripts = 64;
constexpr qsizetype kMaximumMessageBytes = 256 * 1024 * 1024;

template<typename T, typename Error>
T Require(fv::DiscriminatedResult<T, Error> result, const char *operation) {
    if (!result) {
        std::string message = std::string(operation) + " failed";
        if (!result.Error().diagnostic.empty()) {
            message += ": " + result.Error().diagnostic;
        }
        throw std::runtime_error(std::move(message));
    }
    return std::move(result).Value();
}

QString Q(const std::string &text) {
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
}

std::string S(const QString &text) {
    const QByteArray utf8 = text.toUtf8();
    return std::string(utf8.constData(), static_cast<std::size_t>(utf8.size()));
}

double Round(double value, int digits) {
    const double scale = std::pow(10.0, digits);
    return std::round(value * scale) / scale;
}

QJsonArray Vec(const fv::Vector3 &value, int digits = 3) {
    return QJsonArray{Round(value.x, digits), Round(value.y, digits),
                      Round(value.z, digits)};
}

float Length(const fv::Vector3 &value) {
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

struct Euler {
    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;
};

// Yaw about +Y (0 = facing +Z, 90 = facing +X), pitch nose-up positive,
// roll positive when the car's local +X side is lower.
Euler ToEuler(float x, float y, float z, float w) {
    constexpr double degrees = 57.29577951308232;
    const double fx = 2.0 * (x * z + w * y);
    const double fy = 2.0 * (y * z - w * x);
    const double fz = 1.0 - 2.0 * (x * x + y * y);
    const double ry = 2.0 * (x * y + w * z);
    const double uy = 1.0 - 2.0 * (x * x + z * z);
    Euler euler;
    euler.yaw = std::atan2(fx, fz) * degrees;
    euler.pitch = std::asin(std::clamp(fy, -1.0, 1.0)) * degrees;
    euler.roll = std::atan2(-ry, uy) * degrees;
    return euler;
}

QString SettingString(const QJsonValue &value) {
    if (value.isBool()) {
        return value.toBool() ? QStringLiteral("true") : QStringLiteral("false");
    }
    if (value.isDouble()) {
        const double number = value.toDouble();
        if (std::floor(number) == number && std::fabs(number) < 9.0e15) {
            return QString::number(static_cast<qint64>(number));
        }
        return QString::number(number, 'g', 17);
    }
    return value.toString();
}

QJsonObject OptionSettingsJson(const OptionSettings &settings) {
    QJsonObject object;
    for (const auto &[key, value] : settings) object.insert(Q(key), Q(value));
    return object;
}

QJsonObject ToolText(const QJsonObject &payload, bool isError) {
    const QString text = QString::fromUtf8(
            QJsonDocument(payload).toJson(QJsonDocument::Compact));
    QJsonObject result{
            {QStringLiteral("content"),
             QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("text")},
                                    {QStringLiteral("text"), text}}}}};
    if (isError) result.insert(QStringLiteral("isError"), true);
    return result;
}

QJsonObject Error(const QString &message) {
    return QJsonObject{{QStringLiteral("error"), message}};
}

QString DefaultOutputPath(const QString &prefix, const QString &suffix) {
    const QString base = QStandardPaths::writableLocation(
                                 QStandardPaths::DocumentsLocation) +
            QStringLiteral("/ForeverTAS/Claude");
    QDir().mkpath(base);
    return base + '/' + prefix +
            QDateTime::currentDateTime().toString(
                    QStringLiteral("yyyyMMdd-HHmmss-zzz")) +
            suffix;
}

int ClampedInt(const QJsonValue &value, int fallback, int minimum, int maximum) {
    if (!value.isDouble()) return fallback;
    const double number = value.toDouble();
    if (!std::isfinite(number)) return fallback;
    return static_cast<int>(std::clamp<double>(number, minimum, maximum));
}

}  // namespace

// --------------------------------------------------------------------------
// Headless simulation (runs only on simulationPool_'s single thread).
// --------------------------------------------------------------------------

struct ClaudeSimulationCache {
    std::string packs;
    std::string scenario;
    std::uint32_t horizonMs = 0u;
    std::optional<fvx::PhysicsSandbox> sandbox;
    std::optional<fvx::PhysicsSandboxState> initialState;
    fvx::PhysicsSandboxStateView initialView;
    std::vector<fvx::PhysicsSandboxInputEvent> canonicalInputs;
    std::int64_t raceStartMs = 0;
    std::string mapName;

    void ensure(const std::string &packsDirectory,
                const std::string &scenarioPath,
                std::uint32_t simulationHorizonMs) {
        if (sandbox && packs == packsDirectory && scenario == scenarioPath &&
            horizonMs == simulationHorizonMs) {
            return;
        }
        sandbox.reset();
        initialState.reset();
        if (packsDirectory.empty()) {
            throw std::runtime_error("No Packs directory is configured (use load_scenario with packs_directory)");
        }
        if (scenarioPath.empty()) {
            throw std::runtime_error("No replay or challenge is loaded (use load_scenario)");
        }
        const fv::ReplayIdentity identity{scenarioPath};
        fv::AssetSource source = Require(
                fv::OpenInstalledPackDirectory(packsDirectory),
                "opening Packs directory");
        fv::AssetBytes bytes = Require(
                ReadReplayFileUtf8(scenarioPath, identity), "reading scenario");
        fvx::PhysicsSandboxOptions options;
        options.backend = fv::SimulationBackend::OptimizedCpu;
        options.tickDurationMs = kTickMs;
        options.timelineMode = fvx::PhysicsSandboxTimelineMode::Canonical;
        options.simulationHorizonMs = simulationHorizonMs;
        fvx::PhysicsSandbox created = Require(
                fvx::CreatePhysicsSandbox(std::move(source), options),
                "creating simulation");
        initialView = Require(
                created.LoadScenario({bytes.data(), bytes.size()}, identity),
                "loading scenario");
        canonicalInputs = Require(created.ReadInputs(), "reading canonical inputs");
        raceStartMs = 0;
        for (const auto &event : canonicalInputs) {
            if (event.action == fvx::PhysicsSandboxInputAction::RaceRunning) {
                raceStartMs = event.timeMs;
                break;
            }
        }
        auto name = created.ReadMapName();
        mapName = name ? std::move(name).Value() : std::string();
        initialState = Require(created.CaptureState(), "capturing initial state");
        sandbox.emplace(std::move(created));
        packs = packsDirectory;
        scenario = scenarioPath;
        horizonMs = simulationHorizonMs;
    }

    std::int64_t inputTime(std::uint64_t simulationTimeMs) const {
        return UserTimelineTimeFromSimulationTime(
                static_cast<std::int64_t>(simulationTimeMs), kTickMs,
                raceStartMs);
    }
};

namespace {

struct SimulationOptions {
    int sampleIntervalMs = 250;
    std::int64_t fromMs = 0;
    std::int64_t toMs = std::numeric_limits<std::int64_t>::max();
    QString csvPath;
    bool samples = true;
};

const char *const kSampleFields =
        "t_ms,x,y,z,speed_kmh,vx,vy,vz,fwd_speed,side_speed,yaw,pitch,roll,"
        "steer,gas,brake,gear,rpm,wheels_on_ground,sliding,turbo,checkpoints";

QJsonObject SimulateScript(ClaudeSimulationCache &cache,
                           const std::string &script,
                           const SimulationOptions &options) {
    const InputScriptParseResult parsed = ParseInputScript(script);
    if (!parsed) {
        return Error(QStringLiteral("Script parse error: %1").arg(Q(*parsed.error)));
    }
    InputScriptBaselineResult baseline = BuildInputScriptBaseline(
            cache.canonicalInputs, parsed.commands, kTickMs);
    if (!baseline) {
        return Error(QStringLiteral("Script error: %1").arg(Q(*baseline.error)));
    }
    ConvertKeyboardSteeringToAnalog(baseline.events);

    fvx::PhysicsSandbox &sandbox = *cache.sandbox;
    fvx::PhysicsSandboxStateView view = Require(
            sandbox.RestoreState(*cache.initialState), "restoring initial state");
    Require(sandbox.ReplaceInputs(std::move(baseline.events)), "applying inputs");

    std::unique_ptr<QFile> csv;
    std::unique_ptr<QTextStream> csvStream;
    if (!options.csvPath.isEmpty()) {
        QDir().mkpath(QFileInfo(options.csvPath).absolutePath());
        csv = std::make_unique<QFile>(options.csvPath);
        if (!csv->open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
            return Error(QStringLiteral("Cannot write CSV %1").arg(options.csvPath));
        }
        csvStream = std::make_unique<QTextStream>(csv.get());
        *csvStream << "sim_ms," << kSampleFields << ",respawns\n";
    }

    QJsonArray samples;
    QJsonArray checkpoints;
    double maxSpeed = 0.0;
    double maxSpeedAt = 0.0;
    std::uint64_t airborneTicks = 0u;
    std::uint64_t slidingTicks = 0u;
    std::uint64_t ticks = 0u;
    const std::uint64_t tickLimit = cache.horizonMs / kTickMs;

    const auto record = [&](const fvx::PhysicsSandboxStateView &state) {
        const auto &car = state.car;
        const std::int64_t t = cache.inputTime(state.timeMs);
        const double speedKmh = Length(car.linearSpeed) * 3.6;
        int onGround = 0;
        for (bool contact : car.wheelContact) onGround += contact ? 1 : 0;
        const bool sliding = car.sliding;
        if (speedKmh > maxSpeed) {
            maxSpeed = speedKmh;
            maxSpeedAt = static_cast<double>(t);
        }
        if (state.tick > 0u) {
            if (onGround == 0) ++airborneTicks;
            if (sliding) ++slidingTicks;
        }
        for (const auto &event : state.acceptedCheckpointEvents) {
            checkpoints.append(QJsonObject{
                    {QStringLiteral("finish"), event.finish},
                    {QStringLiteral("number"), static_cast<qint64>(event.checkpointIndex) + 1},
                    {QStringLiteral("lap"), static_cast<qint64>(event.lap)},
                    {QStringLiteral("slot"), static_cast<qint64>(event.checkpointSlot)},
                    {QStringLiteral("event_index"), static_cast<qint64>(event.eventIndex)},
                    {QStringLiteral("t_ms"), static_cast<qint64>(cache.inputTime(event.timeMs))},
                    {QStringLiteral("race_time_ms"),
                     static_cast<qint64>(event.timeMs) - cache.raceStartMs},
                    {QStringLiteral("speed_kmh"), Round(speedKmh, 2)}});
        }
        const bool wantSample = options.samples && t >= options.fromMs &&
                t <= options.toMs &&
                (options.sampleIntervalMs <= 0 ? false
                         : t % options.sampleIntervalMs == 0);
        if (!wantSample && !csvStream) return;
        const Euler euler = ToEuler(car.rotationX, car.rotationY,
                                    car.rotationZ, car.rotationW);
        const int steer = static_cast<int>(std::lround(state.steering * 65536.0f));
        const QList<double> values{
                static_cast<double>(t),
                Round(car.position.x, 3), Round(car.position.y, 3), Round(car.position.z, 3),
                Round(speedKmh, 2),
                Round(car.linearSpeed.x, 3), Round(car.linearSpeed.y, 3), Round(car.linearSpeed.z, 3),
                Round(car.localSpeed.z, 3), Round(car.localSpeed.x, 3),
                Round(euler.yaw, 2), Round(euler.pitch, 2), Round(euler.roll, 2),
                static_cast<double>(steer), Round(state.accelerate, 3), Round(state.brake, 3),
                static_cast<double>(car.gear), Round(car.rpm, 1),
                static_cast<double>(onGround), sliding ? 1.0 : 0.0,
                Round(car.turbo, 3), static_cast<double>(state.checkpointsCollected)};
        if (wantSample) {
            QJsonArray row;
            for (double value : values) row.append(value);
            samples.append(row);
        }
        if (csvStream) {
            *csvStream << state.timeMs;
            for (double value : values) *csvStream << ',' << value;
            *csvStream << ',' << state.respawnCount << '\n';
        }
    };

    // A car that stays (almost) still for 10 s after the start will not
    // finish; stop there instead of simulating the whole horizon.
    constexpr std::uint64_t kStallTicks = 10000u / kTickMs;
    std::uint64_t stillTicks = 0u;
    bool stalled = false;
    record(view);
    while (ticks < tickLimit && !view.raceCompleted) {
        view = Require(sandbox.AdvanceTicks(1u), "simulating");
        ++ticks;
        record(view);
        const bool raceStarted = static_cast<std::int64_t>(view.timeMs) > cache.raceStartMs + 1000;
        stillTicks = raceStarted && Length(view.car.linearSpeed) < 0.5f ? stillTicks + 1u : 0u;
        if (stillTicks >= kStallTicks) {
            stalled = true;
            break;
        }
    }
    if (csvStream) csvStream->flush();

    QJsonObject result{
            {QStringLiteral("finished"), view.raceCompleted},
            {QStringLiteral("checkpoints_collected"), static_cast<qint64>(view.checkpointsCollected)},
            {QStringLiteral("checkpoints_total"), static_cast<qint64>(view.checkpointsTotal)},
            {QStringLiteral("laps"), QStringLiteral("%1/%2").arg(view.completedLaps).arg(view.totalLaps)},
            {QStringLiteral("respawns"), static_cast<qint64>(view.respawnCount)},
            {QStringLiteral("simulated_until_ms"), static_cast<qint64>(cache.inputTime(view.timeMs))},
            {QStringLiteral("max_speed_kmh"), Round(maxSpeed, 2)},
            {QStringLiteral("max_speed_at_ms"), maxSpeedAt},
            {QStringLiteral("airborne_ms"), static_cast<qint64>(airborneTicks * kTickMs)},
            {QStringLiteral("sliding_ms"), static_cast<qint64>(slidingTicks * kTickMs)},
            {QStringLiteral("checkpoint_times"), checkpoints}};
    if (view.finishTimeMs) {
        result.insert(QStringLiteral("finish_time_ms"), static_cast<qint64>(*view.finishTimeMs));
    }
    if (view.finishTime && view.finishTime->IsValid()) {
        result.insert(QStringLiteral("finish_time_precise_ms"),
                      static_cast<double>(view.finishTime->estimatedNs) / 1.0e6);
    }
    if (view.stuntsScore) {
        result.insert(QStringLiteral("stunts_score"), static_cast<qint64>(*view.stuntsScore));
    }
    if (!view.raceCompleted) {
        result.insert(QStringLiteral("stalled"), stalled);
        result.insert(QStringLiteral("end_position"), Vec(view.car.position));
        result.insert(QStringLiteral("end_speed_kmh"), Round(Length(view.car.linearSpeed) * 3.6, 2));
    }
    if (options.samples && options.sampleIntervalMs > 0) {
        result.insert(QStringLiteral("sample_fields"), QString::fromLatin1(kSampleFields));
        result.insert(QStringLiteral("samples"), samples);
    }
    if (csv) result.insert(QStringLiteral("csv_path"), QDir::toNativeSeparators(options.csvPath));
    return result;
}

QJsonObject ScenarioInfo(const ClaudeSimulationCache &cache) {
    const auto &view = cache.initialView;
    const Euler euler = ToEuler(view.car.rotationX, view.car.rotationY,
                                view.car.rotationZ, view.car.rotationW);
    return QJsonObject{
            {QStringLiteral("map_name"), Q(cache.mapName)},
            {QStringLiteral("checkpoints"), static_cast<qint64>(view.checkpointsTotal)},
            {QStringLiteral("laps"), static_cast<qint64>(view.totalLaps)},
            {QStringLiteral("environment"), static_cast<int>(view.mapEnvironment)},
            {QStringLiteral("vehicle"), static_cast<int>(view.vehicleModel)},
            {QStringLiteral("start_position"), Vec(view.car.position)},
            {QStringLiteral("start_yaw_degrees"), Round(euler.yaw, 2)}};
}

fv::Vector3 Transform(const fvx::PhysicsSandboxTransform &transform,
                      float x, float y, float z) {
    return {transform.basisX.x * x + transform.basisY.x * y + transform.basisZ.x * z +
                    transform.translation.x,
            transform.basisX.y * x + transform.basisY.y * y + transform.basisZ.y * z +
                    transform.translation.y,
            transform.basisX.z * x + transform.basisY.z * y + transform.basisZ.z * z +
                    transform.translation.z};
}

QString BlockKind(const QString &name) {
    const QString lower = name.toLower();
    const bool start = lower.contains(QStringLiteral("start"));
    const bool finish = lower.contains(QStringLiteral("finish"));
    if (lower.contains(QStringLiteral("multilap")) || (start && finish)) return QStringLiteral("start_finish");
    if (finish) return QStringLiteral("finish");
    if (start) return QStringLiteral("start");
    if (lower.contains(QStringLiteral("checkpoint"))) return QStringLiteral("checkpoint");
    if (lower.contains(QStringLiteral("turbo"))) return QStringLiteral("turbo");
    return {};
}

QJsonObject MapLayout(ClaudeSimulationCache &cache, bool includeBlocks,
                      int maxBlocks) {
    const fvx::PhysicsSandboxRenderSceneHandle scene = Require(
            cache.sandbox->ReadRenderScene(), "reading map scene");
    struct Block {
        QString name;
        fv::Vector3 low{std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max()};
        fv::Vector3 high{-std::numeric_limits<float>::max(),
                         -std::numeric_limits<float>::max(),
                         -std::numeric_limits<float>::max()};
        fv::Vector3 origin{};
        double yaw = 0.0;
        bool trigger = false;
    };
    QHash<quint64, Block> blocks;
    QList<quint64> order;
    for (const auto &instance : scene->instances) {
        const bool trigger =
                instance.purpose == fvx::PhysicsSandboxScenePurpose::CheckpointTrigger;
        if ((instance.purpose != fvx::PhysicsSandboxScenePurpose::PlacedBlock && !trigger) ||
            instance.meshIndex >= scene->meshes.size()) {
            continue;
        }
        const auto &provenance = instance.provenance;
        const quint64 key = provenance.placementIdentity
                ? *provenance.placementIdentity
                : (provenance.blockInstanceId ? *provenance.blockInstanceId : instance.id);
        auto found = blocks.find(key);
        if (found == blocks.end()) {
            Block block;
            block.name = Q(provenance.blockName);
            block.origin = instance.worldTransform.translation;
            const auto &t = instance.worldTransform;
            block.yaw = std::atan2(t.basisZ.x, t.basisZ.z) * 57.29577951308232;
            found = blocks.insert(key, block);
            order.append(key);
        }
        Block &block = found.value();
        block.trigger = block.trigger || trigger;
        const auto &mesh = scene->meshes[instance.meshIndex];
        for (int corner = 0; corner < 8; ++corner) {
            const fv::Vector3 point = Transform(
                    instance.worldTransform,
                    (corner & 1) ? mesh.boundsMax.x : mesh.boundsMin.x,
                    (corner & 2) ? mesh.boundsMax.y : mesh.boundsMin.y,
                    (corner & 4) ? mesh.boundsMax.z : mesh.boundsMin.z);
            block.low = {std::min(block.low.x, point.x), std::min(block.low.y, point.y),
                         std::min(block.low.z, point.z)};
            block.high = {std::max(block.high.x, point.x), std::max(block.high.y, point.y),
                          std::max(block.high.z, point.z)};
        }
    }
    const auto describe = [](const Block &block, const QString &kind) {
        QJsonObject entry{
                {QStringLiteral("block"), block.name},
                {QStringLiteral("center"),
                 QJsonArray{Round((block.low.x + block.high.x) / 2, 1),
                            Round((block.low.y + block.high.y) / 2, 1),
                            Round((block.low.z + block.high.z) / 2, 1)}},
                {QStringLiteral("size"),
                 QJsonArray{Round(block.high.x - block.low.x, 1),
                            Round(block.high.y - block.low.y, 1),
                            Round(block.high.z - block.low.z, 1)}},
                {QStringLiteral("yaw"), Round(block.yaw, 1)}};
        if (!kind.isEmpty()) entry.insert(QStringLiteral("kind"), kind);
        return entry;
    };
    QJsonArray keyBlocks;
    QJsonArray allBlocks;
    for (const quint64 key : std::as_const(order)) {
        const Block &block = blocks[key];
        const QString kind = block.trigger && BlockKind(block.name).isEmpty()
                ? QStringLiteral("checkpoint") : BlockKind(block.name);
        if (!kind.isEmpty()) keyBlocks.append(describe(block, kind));
        if (includeBlocks && allBlocks.size() < maxBlocks) allBlocks.append(describe(block, kind));
    }
    QJsonObject result = ScenarioInfo(cache);
    result.insert(QStringLiteral("key_blocks"), keyBlocks);
    result.insert(QStringLiteral("block_count"), static_cast<qint64>(order.size()));
    if (includeBlocks) result.insert(QStringLiteral("blocks"), allBlocks);
    return result;
}

}  // namespace

// --------------------------------------------------------------------------
// Transport.
// --------------------------------------------------------------------------

ClaudeBridge::ClaudeBridge(SearchController &controller,
                           QQmlApplicationEngine &engine, QObject *parent)
    : QObject(parent),
      controller_(controller),
      engine_(engine),
      cache_(std::make_shared<ClaudeSimulationCache>()) {
    simulationPool_.setMaxThreadCount(1);
    simulationPool_.setExpiryTimeout(-1);
}

ClaudeBridge::~ClaudeBridge() {
    if (server_ != nullptr) server_->close();
    simulationPool_.clear();
    simulationPool_.waitForDone();
}

bool ClaudeBridge::listen() {
    server_ = new QLocalServer(this);
    server_->setSocketOptions(QLocalServer::UserAccessOption);
    const QString name = QString::fromLatin1(kClaudeBridgeSocketName);
    if (!server_->listen(name)) {
        // A stale socket can only exist on Unix; on Windows a second instance
        // simply keeps the first one as the MCP endpoint.
        QLocalServer::removeServer(name);
        if (!server_->listen(name)) {
            std::fprintf(stderr, "ForeverTAS MCP bridge unavailable: %s\n",
                         server_->errorString().toLocal8Bit().constData());
            return false;
        }
    }
    connect(server_, &QLocalServer::newConnection, this,
            &ClaudeBridge::acceptConnections);
    return true;
}

void ClaudeBridge::acceptConnections() {
    while (QLocalSocket *socket = server_->nextPendingConnection()) {
        buffers_.insert(socket, {});
        connect(socket, &QLocalSocket::readyRead, this,
                [this, socket] { readSocket(socket); });
        connect(socket, &QLocalSocket::disconnected, this, [this, socket] {
            buffers_.remove(socket);
            socket->deleteLater();
        });
    }
}

void ClaudeBridge::readSocket(QLocalSocket *socket) {
    QByteArray &buffer = buffers_[socket];
    buffer.append(socket->readAll());
    if (buffer.size() > kMaximumMessageBytes) {
        socket->abort();
        return;
    }
    qsizetype newline = -1;
    while ((newline = buffer.indexOf('\n')) >= 0) {
        const QByteArray line = buffer.left(newline).trimmed();
        buffer.remove(0, newline + 1);
        if (line.isEmpty()) continue;
        QJsonParseError error{};
        const QJsonDocument document = QJsonDocument::fromJson(line, &error);
        const QPointer<QLocalSocket> guarded(socket);
        if (error.error != QJsonParseError::NoError || !document.isObject()) {
            const QJsonObject response{
                    {QStringLiteral("jsonrpc"), QStringLiteral("2.0")},
                    {QStringLiteral("id"), QJsonValue::Null},
                    {QStringLiteral("error"),
                     QJsonObject{{QStringLiteral("code"), -32700},
                                 {QStringLiteral("message"), QStringLiteral("Parse error")}}}};
            socket->write(QJsonDocument(response).toJson(QJsonDocument::Compact) + '\n');
            continue;
        }
        handleMessage(guarded, document.object());
    }
}

void ClaudeBridge::handleMessage(const QPointer<QLocalSocket> &socket,
                                 const QJsonObject &message) {
    const QString method = message.value(QStringLiteral("method")).toString();
    const QJsonValue id = message.value(QStringLiteral("id"));
    const bool notification = !message.contains(QStringLiteral("id"));
    const auto send = [socket, id](const QJsonObject &body) {
        if (!socket) return;
        QJsonObject response{{QStringLiteral("jsonrpc"), QStringLiteral("2.0")},
                             {QStringLiteral("id"), id}};
        for (auto it = body.begin(); it != body.end(); ++it) {
            response.insert(it.key(), it.value());
        }
        socket->write(QJsonDocument(response).toJson(QJsonDocument::Compact) + '\n');
        socket->flush();
    };
    if (notification) return;  // initialized, cancelled, ...
    const QJsonObject params = message.value(QStringLiteral("params")).toObject();
    if (method == QLatin1String("initialize")) {
        send({{QStringLiteral("result"),
               ClaudeInitializeResult(params, QCoreApplication::applicationVersion())}});
    } else if (method == QLatin1String("ping")) {
        send({{QStringLiteral("result"), QJsonObject{}}});
    } else if (method == QLatin1String("tools/list")) {
        send({{QStringLiteral("result"),
               QJsonObject{{QStringLiteral("tools"), ClaudeToolCatalog()}}}});
    } else if (method == QLatin1String("tools/call")) {
        const QString name = params.value(QStringLiteral("name")).toString();
        const QJsonObject arguments = params.value(QStringLiteral("arguments")).toObject();
        callTool(name, arguments, [send](const QJsonObject &payload, bool isError) {
            send({{QStringLiteral("result"), ToolText(payload, isError)}});
        });
    } else {
        send({{QStringLiteral("error"),
               QJsonObject{{QStringLiteral("code"), -32601},
                           {QStringLiteral("message"),
                            QStringLiteral("Method not found: %1").arg(method)}}}});
    }
}

void ClaudeBridge::callTool(const QString &name, const QJsonObject &arguments,
                            Reply reply) {
    using Handler = void (ClaudeBridge::*)(const QJsonObject &, Reply);
    static const QHash<QString, Handler> handlers{
            {QStringLiteral("get_status"), &ClaudeBridge::toolGetStatus},
            {QStringLiteral("list_components"), &ClaudeBridge::toolListComponents},
            {QStringLiteral("get_documentation"), &ClaudeBridge::toolGetDocumentation},
            {QStringLiteral("load_scenario"), &ClaudeBridge::toolLoadScenario},
            {QStringLiteral("get_map_layout"), &ClaudeBridge::toolGetMapLayout},
            {QStringLiteral("get_input_script"), &ClaudeBridge::toolGetInputScript},
            {QStringLiteral("set_input_script"), &ClaudeBridge::toolSetInputScript},
            {QStringLiteral("use_best_as_base"), &ClaudeBridge::toolUseBestAsBase},
            {QStringLiteral("simulate"), &ClaudeBridge::toolSimulate},
            {QStringLiteral("configure_search"), &ClaudeBridge::toolConfigureSearch},
            {QStringLiteral("evaluate_base"), &ClaudeBridge::toolEvaluateBase},
            {QStringLiteral("start_search"), &ClaudeBridge::toolStartSearch},
            {QStringLiteral("wait_for_search"), &ClaudeBridge::toolWaitForSearch},
            {QStringLiteral("stop_search"), &ClaudeBridge::toolStopSearch},
            {QStringLiteral("get_best_result"), &ClaudeBridge::toolGetBestResult},
            {QStringLiteral("get_search_history"), &ClaudeBridge::toolGetSearchHistory},
            {QStringLiteral("save_input_script"), &ClaudeBridge::toolSaveInputScript},
            {QStringLiteral("capture_window"), &ClaudeBridge::toolCaptureWindow}};
    const auto handler = handlers.constFind(name);
    if (handler == handlers.constEnd()) {
        reply(Error(QStringLiteral("Unknown tool: %1").arg(name)), true);
        return;
    }
    try {
        (this->*handler.value())(arguments, reply);
    } catch (const std::exception &error) {
        reply(Error(QString::fromUtf8(error.what())), true);
    }
}

void ClaudeBridge::waitUntil(std::function<bool()> done, int timeoutMs,
                             std::function<void(bool)> finish) {
    if (done()) {
        finish(false);
        return;
    }
    auto *timer = new QTimer(this);
    auto clock = std::make_shared<QElapsedTimer>();
    clock->start();
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, this,
            [timer, clock, timeoutMs, done = std::move(done),
             finish = std::move(finish)]() {
                const bool satisfied = done();
                if (!satisfied && clock->elapsed() < timeoutMs) return;
                timer->stop();
                timer->deleteLater();
                finish(!satisfied);
            });
    timer->start();
}

void ClaudeBridge::runSimulationJob(
        std::function<QJsonObject(ClaudeSimulationCache &)> work, Reply reply) {
    const QPointer<ClaudeBridge> self(this);
    auto cache = cache_;
    simulationPool_.start([self, cache, work = std::move(work),
                           reply = std::move(reply)]() mutable {
        QJsonObject payload;
        bool failed = false;
        try {
            payload = work(*cache);
            failed = payload.contains(QStringLiteral("error")) &&
                    payload.size() == 1;
        } catch (const std::exception &error) {
            payload = Error(QString::fromUtf8(error.what()));
            failed = true;
        } catch (...) {
            payload = Error(QStringLiteral("Unexpected simulation failure"));
            failed = true;
        }
        if (!self) return;
        QMetaObject::invokeMethod(
                self.data(),
                [reply = std::move(reply), payload = std::move(payload), failed] {
                    reply(payload, failed);
                },
                Qt::QueuedConnection);
    });
}

// --------------------------------------------------------------------------
// State helpers.
// --------------------------------------------------------------------------

std::uint32_t ClaudeBridge::configuredHorizonMs() const {
    bool ok = false;
    const qint64 value = controller_.simulationHorizonMs().toLongLong(&ok);
    if (!ok || value <= 0) return kDefaultSimulationHorizonMs;
    return static_cast<std::uint32_t>(std::min<qint64>(value, kMaximumSimulationHorizonMs));
}

QJsonObject ClaudeBridge::configurationObject() const {
    QJsonArray backends;
    for (const QVariant &option : controller_.simulationBackendOptions()) {
        QJsonObject backend = QJsonObject::fromVariantMap(option.toMap());
        const QString id = backend.value(QStringLiteral("id")).toString();
        if (id == QLatin1String("cuda")) {
            backend.insert(QStringLiteral("available"), controller_.cudaAvailable());
            backend.insert(QStringLiteral("status"), controller_.cudaStatusText());
        } else if (id == QLatin1String("vulkan")) {
            backend.insert(QStringLiteral("available"), controller_.vulkanAvailable());
            backend.insert(QStringLiteral("status"), controller_.vulkanStatusText());
        } else if (id == QLatin1String("hip")) {
            backend.insert(QStringLiteral("available"), controller_.hipAvailable());
            backend.insert(QStringLiteral("status"), controller_.hipStatusText());
        } else {
            backend.insert(QStringLiteral("available"), true);
        }
        backends.append(backend);
    }
    return QJsonObject{
            {QStringLiteral("backend"), controller_.simulationBackendId()},
            {QStringLiteral("backend_options"), backends},
            {QStringLiteral("simulation_horizon_ms"), controller_.simulationHorizonMs()},
            {QStringLiteral("required_horizon_ms"), controller_.requiredSimulationHorizonMs()},
            {QStringLiteral("condition_script"), controller_.conditionScript()},
            {QStringLiteral("algorithm"),
             QJsonObject{{QStringLiteral("id"), controller_.searchAlgorithmId()},
                         {QStringLiteral("settings"),
                          QJsonObject::fromVariantMap(controller_.searchAlgorithmSettings())}}},
            {QStringLiteral("modifiers"), QJsonArray::fromVariantList(controller_.modifierPasses())},
            {QStringLiteral("target"),
             QJsonObject{{QStringLiteral("id"), controller_.evaluationTargetId()},
                         {QStringLiteral("settings"),
                          QJsonObject::fromVariantMap(controller_.evaluationTargetSettings())}}},
            {QStringLiteral("cpu_workers"), controller_.cpuWorkerCount()},
            {QStringLiteral("cuda_batch_size"), controller_.cudaParallelSampleCount()},
            {QStringLiteral("cuda_calibration"), controller_.cudaCalibrationEnabled()},
            {QStringLiteral("vulkan_batch_size"), controller_.vulkanParallelSampleCount()},
            {QStringLiteral("cuda_fast_mode"), controller_.cudaSessionSpecializationEnabled()},
            {QStringLiteral("cuda_fast_mode_available"), controller_.cudaFastModeAvailable()},
            {QStringLiteral("randomize_seeds"), controller_.randomizeSeedsOnStart()},
            {QStringLiteral("autorestart"),
             QJsonObject{{QStringLiteral("mode"), controller_.autoRestartMode()},
                         {QStringLiteral("duration"), controller_.autoRestartDuration()},
                         {QStringLiteral("attempts"), controller_.autoRestartAttempts()}}}};
}

namespace {
qint64 ImprovementCount(const QString &summary) {
    static const QRegularExpression pattern(QStringLiteral("Improvements: ([0-9.]+)([kKmM]?)"));
    const QRegularExpressionMatch match = pattern.match(summary);
    if (!match.hasMatch()) return 0;
    double value = match.captured(1).toDouble();
    const QString suffix = match.captured(2).toLower();
    if (suffix == QLatin1String("k")) value *= 1e3;
    if (suffix == QLatin1String("m")) value *= 1e6;
    return static_cast<qint64>(value);
}
}  // namespace

QJsonObject ClaudeBridge::searchProgressObject() const {
    return QJsonObject{
            {QStringLiteral("improvements"), ImprovementCount(controller_.resultText())},
            {QStringLiteral("running"), controller_.running()},
            {QStringLiteral("stopping"), controller_.stopping()},
            {QStringLiteral("status_text"), controller_.statusText()},
            {QStringLiteral("attempts"), controller_.iterationCountRawText()},
            {QStringLiteral("attempts_per_second"), controller_.throughputText()},
            {QStringLiteral("elapsed"), controller_.elapsedText()},
            {QStringLiteral("best_summary"), controller_.resultText()},
            {QStringLiteral("failure"), QJsonObject::fromVariantMap(controller_.failureDiagnostic())}};
}

QJsonObject ClaudeBridge::statusObject(bool includeInputs) const {
    const QString script = controller_.baseInputScript();
    QJsonObject status{
            {QStringLiteral("version"), QCoreApplication::applicationVersion()},
            {QStringLiteral("packs_directory"), controller_.packsDirectory()},
            {QStringLiteral("auto_detected_packs_directory"), controller_.autoDetectedPacksDirectory()},
            {QStringLiteral("scenario_path"), controller_.replayPath()},
            {QStringLiteral("base_script_lines"), script.isEmpty() ? 0 : static_cast<int>(script.count('\n') + 1)},
            {QStringLiteral("base_script_error"), controller_.baseInputScriptError()},
            {QStringLiteral("replay_input_status"), controller_.replayInputStatusText()},
            {QStringLiteral("configuration"), configurationObject()},
            {QStringLiteral("validation_message"), controller_.validationMessage()},
            {QStringLiteral("can_start"), controller_.canStart()},
            {QStringLiteral("search"), searchProgressObject()},
            {QStringLiteral("base_evaluation"),
             QJsonObject{{QStringLiteral("text"), controller_.baseEvaluationText()},
                         {QStringLiteral("result"),
                          QJsonObject::fromVariantMap(controller_.baseEvaluationResult())}}}};
    if (includeInputs) {
        status.insert(QStringLiteral("base_script"), script);
        status.insert(QStringLiteral("best_script"), controller_.bestInputsText());
    }
    return status;
}

// --------------------------------------------------------------------------
// Tools.
// --------------------------------------------------------------------------

void ClaudeBridge::toolGetStatus(const QJsonObject &arguments, Reply reply) {
    reply(statusObject(arguments.value(QStringLiteral("include_inputs")).toBool()), false);
}

void ClaudeBridge::toolListComponents(const QJsonObject &arguments, Reply reply) {
    QJsonArray algorithms;
    for (const auto &registration : SearchAlgorithmRegistry()) {
        algorithms.append(QJsonObject{
                {QStringLiteral("id"), Q(registration.id)},
                {QStringLiteral("name"), Q(registration.displayName)},
                {QStringLiteral("gpu"), registration.supportsGpuBackends},
                {QStringLiteral("multi_threaded_cpu"), registration.supportsMultiThreadedCpu},
                {QStringLiteral("default_settings"), OptionSettingsJson(registration.defaultSettings)}});
    }
    QJsonArray modifiers;
    for (const auto &registration : ModifierRegistry()) {
        modifiers.append(QJsonObject{
                {QStringLiteral("id"), Q(registration.id)},
                {QStringLiteral("name"), Q(registration.displayName)},
                {QStringLiteral("default_settings"), OptionSettingsJson(registration.defaultSettings)}});
    }
    QJsonArray targets;
    for (const auto &registration : EvaluationTargetRegistry()) {
        targets.append(QJsonObject{
                {QStringLiteral("id"), Q(registration.id)},
                {QStringLiteral("name"), Q(registration.displayName)},
                {QStringLiteral("default_settings"), OptionSettingsJson(registration.defaultSettings)}});
    }
    QJsonObject result{{QStringLiteral("algorithms"), algorithms},
                       {QStringLiteral("modifiers"), modifiers},
                       {QStringLiteral("targets"), targets},
                       {QStringLiteral("notes"),
                        QStringLiteral("Modifier settings ending in TimeMs use the input timeline "
                                       "(script clock). maxTimeMode='horizon' (where supported) "
                                       "follows the search end. Steering values in settings are "
                                       "normalised -1..1; scripts use -65536..65536.")}};
    if (arguments.value(QStringLiteral("include_condition_reference")).toBool()) {
        result.insert(QStringLiteral("condition_reference"),
                      QJsonArray::fromVariantList(controller_.conditionReference()));
    }
    reply(result, false);
}

void ClaudeBridge::toolGetDocumentation(const QJsonObject &arguments, Reply reply) {
    static const QHash<QString, QString> files{
            {QStringLiteral("readme"), QStringLiteral("README.md")},
            {QStringLiteral("search_components"), QStringLiteral("SEARCH_COMPONENTS.md")},
            {QStringLiteral("conditions"), QStringLiteral("CONDITIONS.md")},
            {QStringLiteral("checkpoint_time"), QStringLiteral("CHECKPOINT_TIME.md")},
            {QStringLiteral("input_insertion"), QStringLiteral("INPUT_INSERTION.md")},
            {QStringLiteral("gpu_backends"), QStringLiteral("GPU_BACKENDS.md")},
            {QStringLiteral("gpu_safety"), QStringLiteral("GPU_SAFETY.md")}};
    const QString topic = arguments.value(QStringLiteral("topic")).toString();
    const auto file = files.constFind(topic);
    if (file == files.constEnd()) {
        reply(Error(QStringLiteral("Unknown topic. Use one of: %1")
                            .arg(QStringList(files.keys()).join(QStringLiteral(", ")))),
              true);
        return;
    }
    QFile resource(QStringLiteral(":/claude-docs/") + file.value());
    if (!resource.open(QIODevice::ReadOnly)) {
        reply(Error(QStringLiteral("Documentation is not bundled in this build")), true);
        return;
    }
    reply(QJsonObject{{QStringLiteral("topic"), topic},
                      {QStringLiteral("markdown"), QString::fromUtf8(resource.readAll())}},
          false);
}

void ClaudeBridge::toolLoadScenario(const QJsonObject &arguments, Reply reply) {
    if (controller_.running()) {
        reply(Error(QStringLiteral("Stop the running search before loading another scenario")), true);
        return;
    }
    const QString path = QDir::fromNativeSeparators(arguments.value(QStringLiteral("path")).toString());
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        reply(Error(QStringLiteral("Scenario file not found: %1").arg(path)), true);
        return;
    }
    const QString packs = arguments.value(QStringLiteral("packs_directory")).toString();
    if (!packs.isEmpty()) {
        if (!QFileInfo(packs).isDir()) {
            reply(Error(QStringLiteral("Packs directory not found: %1").arg(packs)), true);
            return;
        }
        controller_.setPacksDirectory(QDir::fromNativeSeparators(packs));
    } else if (controller_.packsDirectory().isEmpty() &&
               !controller_.autoDetectedPacksDirectory().isEmpty()) {
        controller_.applyAutoDetectedPacksDirectory();
    }
    controller_.setReplayPath(path);

    const bool extract = arguments.value(QStringLiteral("extract_inputs")).toBool();
    const auto finish = [this, reply, extract](bool timedOut) {
        if (timedOut) {
            reply(Error(QStringLiteral("Timed out extracting replay inputs")), true);
            return;
        }
        const std::string packsDirectory = S(controller_.packsDirectory());
        const std::string scenario = S(controller_.replayPath());
        const std::string script = S(controller_.baseInputScript());
        const QString extraction = controller_.replayInputStatusText();
        const std::uint32_t horizon = std::max(configuredHorizonMs(), kDefaultSimulateHorizonMs);
        runSimulationJob(
                [packsDirectory, scenario, script, extract, extraction, horizon](
                        ClaudeSimulationCache &cache) {
                    cache.ensure(packsDirectory, scenario, horizon);
                    QJsonObject result = ScenarioInfo(cache);
                    result.insert(QStringLiteral("packs_directory"), Q(packsDirectory));
                    result.insert(QStringLiteral("scenario_path"), Q(scenario));
                    if (extract) {
                        result.insert(QStringLiteral("extraction"), extraction);
                        result.insert(QStringLiteral("base_script"), Q(script));
                        SimulationOptions options;
                        options.samples = false;
                        result.insert(QStringLiteral("base_script_result"),
                                      SimulateScript(cache, script, options));
                    }
                    return result;
                },
                reply);
    };
    if (!extract) {
        finish(false);
        return;
    }
    if (!controller_.canExtractReplayInputs()) {
        reply(Error(QStringLiteral("Inputs cannot be extracted from this scenario (standalone challenges have no inputs, or the Packs directory is missing)")),
              true);
        return;
    }
    controller_.extractReplayInputs();
    waitUntil([this] { return !controller_.extractingReplayInputs(); }, 120000, finish);
}

void ClaudeBridge::toolGetMapLayout(const QJsonObject &arguments, Reply reply) {
    const std::string packsDirectory = S(controller_.packsDirectory());
    const std::string scenario = S(controller_.replayPath());
    const bool blocks = arguments.value(QStringLiteral("include_blocks")).toBool();
    const int maxBlocks = ClampedInt(arguments.value(QStringLiteral("max_blocks")), 400, 1, 100000);
    const std::uint32_t horizon = std::max(configuredHorizonMs(), kDefaultSimulateHorizonMs);
    runSimulationJob(
            [=](ClaudeSimulationCache &cache) {
                cache.ensure(packsDirectory, scenario,
                             cache.sandbox && cache.packs == packsDirectory &&
                                             cache.scenario == scenario
                                     ? cache.horizonMs : horizon);
                return MapLayout(cache, blocks, maxBlocks);
            },
            reply);
}

void ClaudeBridge::toolGetInputScript(const QJsonObject &, Reply reply) {
    reply(QJsonObject{{QStringLiteral("script"), controller_.baseInputScript()},
                      {QStringLiteral("error"), controller_.baseInputScriptError()}},
          false);
}

void ClaudeBridge::toolSetInputScript(const QJsonObject &arguments, Reply reply) {
    if (controller_.running()) {
        reply(Error(QStringLiteral("Stop the running search before changing the base script")), true);
        return;
    }
    const QString script = arguments.value(QStringLiteral("script")).toString();
    const InputScriptParseResult parsed = ParseInputScript(S(script));
    controller_.setBaseInputScript(script);
    if (arguments.value(QStringLiteral("preview")).toBool()) controller_.previewBaseInputs();
    QJsonObject result{{QStringLiteral("applied"), true},
                       {QStringLiteral("commands"), static_cast<qint64>(parsed.commands.size())},
                       {QStringLiteral("parse_error"), parsed ? QString() : Q(*parsed.error)},
                       {QStringLiteral("validation_message"), controller_.validationMessage()}};
    reply(result, !parsed);
}

void ClaudeBridge::toolUseBestAsBase(const QJsonObject &arguments, Reply reply) {
    if (controller_.running()) {
        reply(Error(QStringLiteral("Stop the running search first (stop_search)")), true);
        return;
    }
    const QString source = arguments.value(QStringLiteral("source")).toString(QStringLiteral("best"));
    const QString script = source == QLatin1String("selected_cycle")
            ? controller_.selectedInputsText()
            : controller_.bestInputsText();
    if (script.trimmed().isEmpty()) {
        reply(Error(QStringLiteral("There is no %1 script yet").arg(source)), true);
        return;
    }
    controller_.setBaseInputScript(script);
    reply(QJsonObject{{QStringLiteral("applied"), true},
                      {QStringLiteral("source"), source},
                      {QStringLiteral("base_script_error"), controller_.baseInputScriptError()},
                      {QStringLiteral("lines"), static_cast<int>(script.count('\n') + 1)}},
          false);
}

void ClaudeBridge::toolSimulate(const QJsonObject &arguments, Reply reply) {
    std::vector<std::string> scripts;
    const bool compare = arguments.value(QStringLiteral("scripts")).isArray();
    if (compare) {
        const QJsonArray list = arguments.value(QStringLiteral("scripts")).toArray();
        if (list.isEmpty() || list.size() > kMaximumCompareScripts) {
            reply(Error(QStringLiteral("'scripts' must hold 1..64 scripts")), true);
            return;
        }
        for (const QJsonValue &value : list) scripts.push_back(S(value.toString()));
    } else {
        scripts.push_back(S(arguments.contains(QStringLiteral("script"))
                                    ? arguments.value(QStringLiteral("script")).toString()
                                    : controller_.baseInputScript()));
    }
    const QString scenarioOverride = arguments.value(QStringLiteral("replay_path")).toString();
    const std::string packsDirectory = S(controller_.packsDirectory());
    const std::string scenario = S(scenarioOverride.isEmpty()
                                           ? controller_.replayPath()
                                           : QDir::fromNativeSeparators(scenarioOverride));
    std::uint32_t horizon = kDefaultSimulateHorizonMs;
    if (arguments.value(QStringLiteral("horizon_ms")).isDouble()) {
        const double requested = arguments.value(QStringLiteral("horizon_ms")).toDouble();
        if (requested < kTickMs || requested > kMaximumSimulationHorizonMs) {
            reply(Error(QStringLiteral("horizon_ms out of range")), true);
            return;
        }
        horizon = static_cast<std::uint32_t>(requested) / kTickMs * kTickMs;
    }
    SimulationOptions options;
    options.sampleIntervalMs = ClampedInt(arguments.value(QStringLiteral("sample_interval_ms")), 250, 0, 3600000);
    if (options.sampleIntervalMs % static_cast<int>(kTickMs) != 0) {
        reply(Error(QStringLiteral("sample_interval_ms must be a multiple of 10")), true);
        return;
    }
    options.fromMs = ClampedInt(arguments.value(QStringLiteral("from_ms")), 0, 0, std::numeric_limits<int>::max());
    options.toMs = arguments.value(QStringLiteral("to_ms")).isDouble()
            ? static_cast<std::int64_t>(arguments.value(QStringLiteral("to_ms")).toDouble())
            : std::numeric_limits<std::int64_t>::max();
    QString csv = arguments.value(QStringLiteral("csv_path")).toString();
    if (csv == QLatin1String("auto")) csv = DefaultOutputPath(QStringLiteral("sim-"), QStringLiteral(".csv"));
    options.csvPath = compare ? QString() : QDir::fromNativeSeparators(csv);
    options.samples = !compare;
    runSimulationJob(
            [packsDirectory, scenario, horizon, scripts = std::move(scripts), options, compare](
                    ClaudeSimulationCache &cache) {
                cache.ensure(packsDirectory, scenario, horizon);
                if (!compare) return SimulateScript(cache, scripts.front(), options);
                QJsonArray results;
                for (std::size_t index = 0; index < scripts.size(); ++index) {
                    QJsonObject entry = SimulateScript(cache, scripts[index], options);
                    entry.insert(QStringLiteral("index"), static_cast<qint64>(index));
                    results.append(entry);
                }
                return QJsonObject{{QStringLiteral("results"), results}};
            },
            reply);
}

void ClaudeBridge::toolConfigureSearch(const QJsonObject &arguments, Reply reply) {
    if (controller_.running()) {
        reply(Error(QStringLiteral("Stop the running search before reconfiguring it")), true);
        return;
    }
    QJsonArray problems;
    const auto note = [&problems](const QString &text) { problems.append(text); };
    const auto applySettings = [&](const QJsonObject &requested,
                                   const std::function<void(const QString &, const QString &)> &set,
                                   const std::function<QVariantMap()> &read,
                                   const QString &label) {
        for (auto it = requested.begin(); it != requested.end(); ++it) {
            const QString value = SettingString(it.value());
            set(it.key(), value);
            const QVariantMap current = read();
            if (!current.contains(it.key())) {
                note(QStringLiteral("%1: unknown setting '%2'").arg(label, it.key()));
            } else if (current.value(it.key()).toString() != value) {
                note(QStringLiteral("%1: '%2' kept '%3' (rejected '%4')")
                             .arg(label, it.key(), current.value(it.key()).toString(), value));
            }
        }
    };

    if (arguments.contains(QStringLiteral("backend"))) {
        const QString backend = arguments.value(QStringLiteral("backend")).toString();
        controller_.setSimulationBackendId(backend);
        if (controller_.simulationBackendId() != backend) {
            note(QStringLiteral("backend '%1' is not available in this build").arg(backend));
        }
    }
    if (arguments.contains(QStringLiteral("simulation_horizon_ms"))) {
        controller_.setSimulationHorizonMs(SettingString(arguments.value(QStringLiteral("simulation_horizon_ms"))));
    }
    if (arguments.contains(QStringLiteral("condition_script"))) {
        controller_.setConditionScript(arguments.value(QStringLiteral("condition_script")).toString());
    }
    if (arguments.value(QStringLiteral("algorithm")).isObject()) {
        const QJsonObject algorithm = arguments.value(QStringLiteral("algorithm")).toObject();
        if (algorithm.contains(QStringLiteral("id"))) {
            const QString id = algorithm.value(QStringLiteral("id")).toString();
            controller_.setSearchAlgorithmId(id);
            if (!FindSearchAlgorithm(S(id))) note(QStringLiteral("unknown algorithm '%1'").arg(id));
        }
        applySettings(algorithm.value(QStringLiteral("settings")).toObject(),
                      [this](const QString &k, const QString &v) { controller_.setSearchAlgorithmSetting(k, v); },
                      [this] { return controller_.searchAlgorithmSettings(); },
                      QStringLiteral("algorithm"));
    }
    if (arguments.value(QStringLiteral("target")).isObject()) {
        const QJsonObject target = arguments.value(QStringLiteral("target")).toObject();
        if (target.contains(QStringLiteral("id"))) {
            const QString id = target.value(QStringLiteral("id")).toString();
            controller_.setEvaluationTargetId(id);
            if (!FindEvaluationTarget(S(id))) note(QStringLiteral("unknown target '%1'").arg(id));
        }
        applySettings(target.value(QStringLiteral("settings")).toObject(),
                      [this](const QString &k, const QString &v) { controller_.setEvaluationTargetSetting(k, v); },
                      [this] { return controller_.evaluationTargetSettings(); },
                      QStringLiteral("target"));
    }
    if (arguments.value(QStringLiteral("modifiers")).isArray()) {
        const QJsonArray passes = arguments.value(QStringLiteral("modifiers")).toArray();
        for (int index = static_cast<int>(controller_.modifierPasses().size()) - 1; index >= 0; --index) {
            controller_.removeModifierPass(index);
        }
        for (const QJsonValue &value : passes) {
            const QJsonObject pass = value.toObject();
            const QString id = pass.value(QStringLiteral("id")).toString();
            const int before = static_cast<int>(controller_.modifierPasses().size());
            controller_.addModifierPass(id);
            const int index = static_cast<int>(controller_.modifierPasses().size()) - 1;
            if (index < before) {
                note(QStringLiteral("unknown modifier '%1'").arg(id));
                continue;
            }
            if (pass.contains(QStringLiteral("enabled"))) {
                controller_.setModifierPassEnabled(index, pass.value(QStringLiteral("enabled")).toBool());
            }
            applySettings(pass.value(QStringLiteral("settings")).toObject(),
                          [this, index](const QString &k, const QString &v) {
                              controller_.setModifierPassSetting(index, k, v);
                          },
                          [this, index] {
                              const QVariantMap entry = controller_.modifierPasses().value(index).toMap();
                              return entry.value(QStringLiteral("settings")).toMap();
                          },
                          QStringLiteral("modifier %1 (%2)").arg(index).arg(id));
        }
    }
    if (arguments.contains(QStringLiteral("cpu_workers"))) {
        controller_.setCpuWorkerCount(SettingString(arguments.value(QStringLiteral("cpu_workers"))));
    }
    if (arguments.contains(QStringLiteral("gpu_batch_size"))) {
        const QString size = SettingString(arguments.value(QStringLiteral("gpu_batch_size")));
        controller_.setCudaParallelSampleCount(size);
        controller_.setVulkanParallelSampleCount(size);
        controller_.setHipParallelSampleCount(size);
    }
    if (arguments.contains(QStringLiteral("gpu_calibration"))) {
        const bool enabled = arguments.value(QStringLiteral("gpu_calibration")).toBool();
        controller_.setCudaCalibrationEnabled(enabled);
        controller_.setVulkanCalibrationEnabled(enabled);
        controller_.setHipCalibrationEnabled(enabled);
    }
    if (arguments.contains(QStringLiteral("cuda_fast_mode"))) {
        controller_.setCudaSessionSpecializationEnabled(arguments.value(QStringLiteral("cuda_fast_mode")).toBool());
    }
    if (arguments.contains(QStringLiteral("randomize_seeds"))) {
        controller_.setRandomizeSeedsOnStart(arguments.value(QStringLiteral("randomize_seeds")).toBool());
    }
    if (arguments.value(QStringLiteral("autorestart")).isObject()) {
        const QJsonObject restart = arguments.value(QStringLiteral("autorestart")).toObject();
        if (restart.contains(QStringLiteral("mode")))
            controller_.setAutoRestartMode(restart.value(QStringLiteral("mode")).toString());
        if (restart.contains(QStringLiteral("duration")))
            controller_.setAutoRestartDuration(restart.value(QStringLiteral("duration")).toString());
        if (restart.contains(QStringLiteral("attempts")))
            controller_.setAutoRestartAttempts(SettingString(restart.value(QStringLiteral("attempts"))));
    }
    if (arguments.value(QStringLiteral("extend_horizon")).toBool()) {
        controller_.extendSimulationHorizon();
    }
    reply(QJsonObject{{QStringLiteral("configuration"), configurationObject()},
                      {QStringLiteral("problems"), problems},
                      {QStringLiteral("validation_message"), controller_.validationMessage()},
                      {QStringLiteral("can_start"), controller_.canStart()}},
          false);
}

void ClaudeBridge::toolEvaluateBase(const QJsonObject &arguments, Reply reply) {
    if (!controller_.canEvaluateBase()) {
        reply(Error(QStringLiteral("Base evaluation is not possible now: %1")
                            .arg(controller_.running() ? QStringLiteral("a search is running")
                                                       : controller_.validationMessage())),
              true);
        return;
    }
    const int timeoutMs = static_cast<int>(
            std::clamp(arguments.value(QStringLiteral("timeout_seconds")).toDouble(60.0), 1.0, 290.0) * 1000.0);
    controller_.evaluateBase();
    waitUntil([this] { return !controller_.evaluatingBase(); }, timeoutMs,
              [this, reply](bool timedOut) {
                  reply(QJsonObject{{QStringLiteral("timed_out"), timedOut},
                                    {QStringLiteral("text"), controller_.baseEvaluationText()},
                                    {QStringLiteral("result"),
                                     QJsonObject::fromVariantMap(controller_.baseEvaluationResult())}},
                        false);
              });
}

void ClaudeBridge::toolStartSearch(const QJsonObject &, Reply reply) {
    if (controller_.running()) {
        reply(Error(QStringLiteral("A search is already running")), true);
        return;
    }
    if (!controller_.canStart()) {
        reply(Error(QStringLiteral("Cannot start: %1").arg(controller_.validationMessage())), true);
        return;
    }
    controller_.startSearch();
    waitUntil([this] { return controller_.running(); }, 2000, [this, reply](bool) {
        reply(QJsonObject{{QStringLiteral("started"), controller_.running()},
                          {QStringLiteral("search"), searchProgressObject()},
                          {QStringLiteral("configuration"), configurationObject()}},
              !controller_.running());
    });
}

void ClaudeBridge::toolWaitForSearch(const QJsonObject &arguments, Reply reply) {
    const int timeoutMs = static_cast<int>(
            std::clamp(arguments.value(QStringLiteral("timeout_seconds")).toDouble(30.0), 0.0, 290.0) * 1000.0);
    const bool untilFinished = arguments.value(QStringLiteral("until")).toString() == QLatin1String("finished");
    const qint64 initialImprovements = ImprovementCount(controller_.resultText());
    waitUntil(
            [this, untilFinished, initialImprovements] {
                if (!controller_.running()) return true;
                return !untilFinished &&
                        ImprovementCount(controller_.resultText()) > initialImprovements;
            },
            timeoutMs,
            [this, reply, initialImprovements](bool timedOut) {
                QJsonObject result = searchProgressObject();
                result.insert(QStringLiteral("timed_out"), timedOut);
                result.insert(QStringLiteral("improved"),
                              ImprovementCount(controller_.resultText()) > initialImprovements);
                reply(result, false);
            });
}

void ClaudeBridge::toolStopSearch(const QJsonObject &arguments, Reply reply) {
    if (!controller_.running()) {
        reply(QJsonObject{{QStringLiteral("stopped"), true},
                          {QStringLiteral("search"), searchProgressObject()}},
              false);
        return;
    }
    if (arguments.value(QStringLiteral("abort")).toBool()) controller_.abortSearch();
    else controller_.stopSearch();
    const int timeoutMs = static_cast<int>(
            std::clamp(arguments.value(QStringLiteral("wait_seconds")).toDouble(30.0), 0.0, 290.0) * 1000.0);
    waitUntil([this] { return !controller_.running(); }, timeoutMs, [this, reply](bool timedOut) {
        reply(QJsonObject{{QStringLiteral("stopped"), !controller_.running()},
                          {QStringLiteral("timed_out"), timedOut},
                          {QStringLiteral("search"), searchProgressObject()}},
              false);
    });
}

void ClaudeBridge::toolGetBestResult(const QJsonObject &arguments, Reply reply) {
    const QString best = controller_.bestInputsText();
    QJsonObject result = searchProgressObject();
    result.insert(QStringLiteral("best_script"), best);
    if (best.trimmed().isEmpty() ||
        !arguments.value(QStringLiteral("simulate")).toBool(true)) {
        reply(result, false);
        return;
    }
    const std::string packsDirectory = S(controller_.packsDirectory());
    const std::string scenario = S(controller_.replayPath());
    const std::string script = S(best);
    const std::string base = S(controller_.baseInputScript());
    runSimulationJob(
            [=](ClaudeSimulationCache &cache) {
                cache.ensure(packsDirectory, scenario,
                             cache.sandbox && cache.packs == packsDirectory &&
                                             cache.scenario == scenario
                                     ? cache.horizonMs : kDefaultSimulateHorizonMs);
                SimulationOptions options;
                options.samples = false;
                QJsonObject withSimulation = result;
                withSimulation.insert(QStringLiteral("simulation"), SimulateScript(cache, script, options));
                if (base != script) {
                    withSimulation.insert(QStringLiteral("base_simulation"),
                                          SimulateScript(cache, base, options));
                }
                return withSimulation;
            },
            reply);
}

void ClaudeBridge::toolGetSearchHistory(const QJsonObject &arguments, Reply reply) {
    if (arguments.value(QStringLiteral("session_index")).isDouble()) {
        controller_.selectSession(arguments.value(QStringLiteral("session_index")).toInt());
    }
    if (arguments.value(QStringLiteral("cycle_index")).isDouble()) {
        controller_.selectCycle(arguments.value(QStringLiteral("cycle_index")).toInt());
    }
    reply(QJsonObject{{QStringLiteral("sessions"), QJsonArray::fromVariantList(controller_.sessionOptions())},
                      {QStringLiteral("selected_session"), controller_.selectedSessionDirectory()},
                      {QStringLiteral("cycles"), QJsonArray::fromVariantList(controller_.cycleRows())},
                      {QStringLiteral("selected_cycle_restart"), controller_.selectedCycleRestart()},
                      {QStringLiteral("selected_cycle_script"), controller_.selectedInputsText()}},
          false);
}

void ClaudeBridge::toolSaveInputScript(const QJsonObject &arguments, Reply reply) {
    const QString path = arguments.value(QStringLiteral("path")).toString();
    QString script;
    if (arguments.contains(QStringLiteral("script"))) {
        script = arguments.value(QStringLiteral("script")).toString();
    } else {
        const QString source = arguments.value(QStringLiteral("source")).toString(QStringLiteral("best"));
        script = source == QLatin1String("base") ? controller_.baseInputScript()
                : source == QLatin1String("selected_cycle") ? controller_.selectedInputsText()
                                                            : controller_.bestInputsText();
    }
    if (path.isEmpty() || script.trimmed().isEmpty()) {
        reply(Error(QStringLiteral("Need a path and a non-empty script")), true);
        return;
    }
    QDir().mkpath(QFileInfo(path).absolutePath());
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text) ||
        file.write(script.toUtf8()) < 0 || !file.commit()) {
        reply(Error(QStringLiteral("Could not write %1").arg(path)), true);
        return;
    }
    reply(QJsonObject{{QStringLiteral("saved"), QDir::toNativeSeparators(path)},
                      {QStringLiteral("lines"), static_cast<int>(script.count('\n') + 1)}},
          false);
}

void ClaudeBridge::toolCaptureWindow(const QJsonObject &arguments, Reply reply) {
    auto *window = qobject_cast<QQuickWindow *>(engine_.rootObjects().value(0));
    if (window == nullptr) {
        reply(Error(QStringLiteral("The main window is not available")), true);
        return;
    }
    QString path = arguments.value(QStringLiteral("path")).toString();
    if (path.isEmpty()) path = DefaultOutputPath(QStringLiteral("window-"), QStringLiteral(".png"));
    QDir().mkpath(QFileInfo(path).absolutePath());
    const QImage image = window->grabWindow();
    if (image.isNull() || !image.save(path, "PNG")) {
        reply(Error(QStringLiteral("Could not capture the window to %1").arg(path)), true);
        return;
    }
    reply(QJsonObject{{QStringLiteral("path"), QDir::toNativeSeparators(path)},
                      {QStringLiteral("width"), image.width()},
                      {QStringLiteral("height"), image.height()}},
          false);
}

}  // namespace forevertas::app
