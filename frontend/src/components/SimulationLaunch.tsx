import { useEffect, useRef, useState } from "react";
import { AlertCircle, GitBranch, Loader2 } from "lucide-react";
import { api } from "../api";
import type { Artifact, SimulationJob } from "../types";
import { useStore } from "../store";

const STAGE_CN: Record<string, string> = {
  queued: "等待执行", compiling_spec: "整理证据与参与方", validating: "检查证据",
  preparing_direct: "准备参与方", building_graph: "整理参与方关系",
  retrying_ontology: "参与方整理暂时失败，正在重试", resuming: "从保存进度继续",
  simulating: "多方行动推演", compiling_scenarios: "整理情景与观察条件",
  completed: "推演完成", partial: "部分完成", failed: "推演失败",
  cancel_requested: "正在停止", cancelled: "已取消",
};
const RUNNING = new Set(["queued", "running", "cancelling"]);

function friendlyError(reason: unknown) {
  const message = reason instanceof Error ? reason.message : String(reason || "");
  if (message.includes("Ontology generation failed")) return "参与方关系整理暂时失败，可以从保存进度继续，或重新推演。";
  if (message.includes("safety budget is exhausted")) return "本次模型调用额度已用完。服务额度恢复后可继续，证据图和已保存进度仍在。";
  if (message.includes("ZEP read quota") || message.toLowerCase().includes("rate limit")) return "服务暂时达到调用上限，请稍后重试。已提交的进度会保留。";
  if (message.includes("durable workflow checkpoint")) return "这次任务还没有可恢复的进度，请选择重新推演。";
  return message;
}

export default function SimulationLaunch({ artifact }: { artifact: Artifact }) {
  const caseId = useStore((state) => state.currentCaseId);
  return caseId ? <SimulationLaunchPanel key={`${caseId}:${artifact.id}`} artifact={artifact} caseId={caseId} /> : null;
}

function SimulationLaunchPanel({ artifact, caseId }: { artifact: Artifact; caseId: string }) {
  const loadCase = useStore((state) => state.loadCase);
  const selectArtifact = useStore((state) => state.selectArtifact);
  const [job, setJob] = useState<SimulationJob | null>(null);
  const [restoring, setRestoring] = useState(true);
  const [error, setError] = useState("");
  const [connectionNotice, setConnectionNotice] = useState("");
  const [pending, setPending] = useState<"start" | "resume" | "cancel" | null>(null);
  const pendingRef = useRef(false);
  const alive = useRef(true);
  const [actorCap, setActorCap] = useState("auto");
  const [horizon, setHorizon] = useState(30);
  const [previewRevision, setPreviewRevision] = useState(0);
  const [preview, setPreview] = useState<Awaited<ReturnType<typeof api.simulationPreview>> | null>(null);
  const [previewLoading, setPreviewLoading] = useState(true);
  const [previewError, setPreviewError] = useState("");
  const body = () => ({
    source_graph_artifact_id: artifact.id, question: artifact.payload?.question,
    horizon_days: horizon, mode: "quick" as const,
    ...(actorCap === "auto" ? {} : { max_actors: Number(actorCap) }),
  });

  useEffect(() => {
    alive.current = true;
    let active = true;
    void api.simulations(caseId).then((jobs) => {
      if (active) setJob(jobs.find((item) => item.graph_artifact_id === artifact.id) ?? null);
    }).catch(() => {
      if (active) setError("暂时未能读取上一次任务；启动时会自动检查是否已有相同任务。");
    }).finally(() => { if (active) setRestoring(false); });
    return () => { active = false; alive.current = false; };
  }, [artifact.id, caseId]);

  useEffect(() => {
    let active = true;
    setPreview(null);
    setPreviewError("");
    setPreviewLoading(true);
    void api.simulationPreview(caseId, {
      source_graph_artifact_id: artifact.id, question: artifact.payload?.question,
      horizon_days: horizon, mode: "quick",
      ...(actorCap === "auto" ? {} : { max_actors: Number(actorCap) }),
    }).then((result) => { if (active) setPreview(result); })
      .catch((reason) => { if (active) setPreviewError(friendlyError(reason)); })
      .finally(() => { if (active) setPreviewLoading(false); });
    return () => { active = false; };
  }, [actorCap, horizon, previewRevision, artifact.id, artifact.payload?.question, caseId]);

  useEffect(() => {
    if (!job || !RUNNING.has(job.status)) return;
    let active = true;
    let timer: number;
    const jobId = job.id;
    const poll = async () => {
      let again = true;
      try {
        const next = await api.simulation(jobId);
        if (!active) return;
        setJob(next);
        setConnectionNotice("");
        again = RUNNING.has(next.status);
        if (next.artifact_id) {
          await loadCase(caseId);
          if (active) selectArtifact(next.artifact_id);
        }
      } catch {
        if (active) setConnectionNotice("进度连接暂时中断，正在自动重试。后台推演会继续。");
      } finally {
        if (active && again) timer = window.setTimeout(() => void poll(), 1800);
      }
    };
    timer = window.setTimeout(() => void poll(), 1200);
    return () => { active = false; window.clearTimeout(timer); };
  }, [job?.id, job?.status, caseId, loadCase, selectArtifact]);

  const act = async (action: "start" | "resume" | "cancel") => {
    if (pendingRef.current) return;
    pendingRef.current = true;
    setPending(action);
    setError("");
    try {
      const next = action === "cancel" && job ? await api.cancelSimulation(job.id)
        : action === "resume" && job ? await api.resumeSimulation(job.id)
        : await api.startSimulation(caseId, { ...body(), rerun: Boolean(job && !RUNNING.has(job.status)) });
      if (!alive.current) return;
      setJob(next);
      if (next.artifact_id) {
        await loadCase(caseId);
        if (alive.current) selectArtifact(next.artifact_id);
      }
    } catch (reason) {
      if (alive.current) setError(friendlyError(reason));
    } finally {
      pendingRef.current = false;
      if (alive.current) setPending(null);
    }
  };

  const busy = job && RUNNING.has(job.status);
  const canStart = !pending && !restoring && !previewLoading && Boolean(preview) && !previewError;
  const progress = Math.max(0, Math.min(100, Math.round((job?.progress || 0) * 100)));
  return <div className="mb-3 rounded-card border border-jade/25 bg-jade-soft/50 p-3.5">
    <div className="flex items-start gap-3">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-jade text-white"><GitBranch size={15} /></span>
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-semibold text-ink">事件预测员 · 多方事件推演</p>
        <p className="mt-1 text-[11.5px] leading-relaxed text-mute">根据这张证据图，推演各方可能的行动，形成条件情景和后续观察清单。通常需要数分钟，可在后台继续。</p>
        {!busy && <div className="mt-3 rounded-lg border border-jade/20 bg-white/60 p-2.5">
          <div className="flex flex-wrap gap-x-4 gap-y-2">
            <label className="flex items-center gap-2 text-[11px] text-ink">参与方
              <select aria-label="参与方上限" value={actorCap} disabled={Boolean(pending)} onChange={(event) => setActorCap(event.target.value)} className="rounded-md border border-edge bg-card px-2 py-1 text-[11px] outline-none focus:border-jade">
                <option value="auto">自动推荐</option>{[4, 6, 8, 10].map((count) => <option key={count} value={count}>最多 {count} 个</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2 text-[11px] text-ink">观察窗口
              <select aria-label="观察窗口" value={horizon} disabled={Boolean(pending)} onChange={(event) => setHorizon(Number(event.target.value))} className="rounded-md border border-edge bg-card px-2 py-1 text-[11px] outline-none focus:border-jade">
                {[7, 30, 90].map((days) => <option key={days} value={days}>未来 {days} 天</option>)}
              </select>
            </label>
          </div>
          {previewLoading && <p role="status" className="mt-2 flex items-center gap-1.5 text-[11px] text-mute"><Loader2 size={12} className="animate-spin" />正在预览参与方…</p>}
          {previewError && <div className="mt-2 text-[11px] text-[#A33A32]"><p role="alert">{previewError}</p><button onClick={() => setPreviewRevision((value) => value + 1)} className="mt-1 underline">重新预览</button></div>}
          {preview && <div className="mt-2 text-[11px] leading-relaxed text-mute">
            <p>{preview.actors.length} 个参与方{preview.evidence_count !== undefined ? ` · ${preview.evidence_count} 条证据` : ""} · 预览不调用模型</p>
            <div className="mt-1.5 flex flex-wrap gap-1">{preview.actors.map((actor) => <span key={actor.id} title={actor.focus || actor.selection_reason} className="rounded-full bg-jade-soft px-2 py-0.5 text-jade">{actor.label}</span>)}</div>
            <details className="mt-2"><summary className="cursor-pointer text-faint">查看各方关注点与入选理由</summary><ul className="mt-1.5 space-y-2">{preview.actors.map((actor) => <li key={actor.id}><p className="text-ink">{actor.label}：{actor.focus || actor.selection_reason}</p>{actor.focus && <p className="text-faint">{actor.selection_reason}</p>}</li>)}</ul><p className="mt-2 text-faint">关注点是角色建模假设，仍需结合事件证据复核。</p></details>
            {preview.notices?.map((notice, index) => <p key={index} className="mt-2 text-[#775B19]">{notice}</p>)}
          </div>}
        </div>}
        {busy ? <div className="mt-3">
          <p className="mb-2 text-[11px] leading-relaxed text-mute">可以继续聊天或浏览其他资料，完成后结果会出现在侧边栏。</p>
          <div className="mb-1.5 flex items-center justify-between text-[11px] text-mute"><span className="flex items-center gap-1.5"><Loader2 size={12} className="animate-spin" />{STAGE_CN[job.stage] ?? "正在推演"}</span><span>{progress}%</span></div>
          <div className="h-1.5 overflow-hidden rounded-full bg-edge"><div className="h-full rounded-full bg-jade transition-all" style={{ width: `${Math.max(3, progress)}%` }} /></div>
          {connectionNotice && <p role="status" className="mt-2 text-[11px] text-[#775B19]">{connectionNotice}</p>}
          <button disabled={Boolean(pending) || job.status === "cancelling"} onClick={() => void act("cancel")} className="mt-2 text-[11px] text-mute underline disabled:cursor-not-allowed disabled:text-faint">{pending === "cancel" || job.status === "cancelling" ? "正在停止…" : "取消推演"}</button>
        </div> : <>
          {job?.status === "failed" && <p className="mt-3 text-[11px] leading-relaxed text-mute">上一次推演未完成。可恢复原任务；需要使用当前参与方和观察窗口时，请重新推演。</p>}
          <div className="mt-3 flex flex-wrap gap-2">
            {job?.status === "failed" && <button disabled={Boolean(pending)} onClick={() => void act("resume")} className="rounded-lg border border-jade/30 px-3 py-1.5 text-[12px] font-medium text-jade disabled:opacity-50">{pending === "resume" ? "正在恢复…" : "从保存进度继续"}</button>}
            <button disabled={!canStart} onClick={() => void act("start")} className="rounded-lg bg-jade px-3 py-1.5 text-[12px] font-medium text-white hover:bg-[#0c665f] disabled:cursor-not-allowed disabled:opacity-50">{pending === "start" ? "正在启动…" : restoring ? "正在恢复任务…" : job ? "按当前设置重新推演" : "开始推演"}</button>
            {job?.artifact_id && <button onClick={() => selectArtifact(job.artifact_id!)} className="px-2 py-1.5 text-[12px] text-jade underline">查看上次结果</button>}
          </div>
        </>}
        {(error || job?.error) && <p role="alert" className="mt-2 flex items-start gap-1.5 break-words text-[11px] text-[#A33A32]"><AlertCircle size={12} className="mt-0.5 shrink-0" />{error || friendlyError(job?.error)}</p>}
      </div>
    </div>
  </div>;
}
