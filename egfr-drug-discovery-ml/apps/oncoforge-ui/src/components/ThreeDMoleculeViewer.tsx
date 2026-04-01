import { memo, useEffect, useRef, useState } from "react";

type RenderStyle = "ball-stick" | "line";

interface ThreeDMoleculeViewerProps {
  molBlock?: string | null;
  renderStyle: RenderStyle;
  heightClassName?: string;
  emptyMessage?: string;
  caption?: string;
}

export const ThreeDMoleculeViewer = memo(function ThreeDMoleculeViewer({
  molBlock,
  renderStyle,
  heightClassName = "h-[420px]",
  emptyMessage = "Vizualizarea 3D va aparea aici dupa selectia unei molecule.",
  caption,
}: ThreeDMoleculeViewerProps) {
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!molBlock || !viewerRef.current) {
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    const mount = viewerRef.current;

    async function render3D() {
      setLoading(true);
      setError(null);

      try {
        const $3Dmol = await import("3dmol");
        if (cancelled || !mount) {
          return;
        }

        mount.innerHTML = "";
        const viewer = $3Dmol.createViewer(mount, { backgroundColor: "#06111d" });
        viewer.clear();
        viewer.addModel(molBlock, "mol");

        if (renderStyle === "ball-stick") {
          viewer.setStyle({}, { stick: { radius: 0.18, colorscheme: "cyanCarbon" }, sphere: { scale: 0.28 } });
        } else {
          viewer.setStyle({}, { line: { colorscheme: "cyanCarbon", linewidth: 2.5 } });
        }

        viewer.zoomTo();
        viewer.render();
      } catch {
        if (!cancelled) {
          setError("Vizualizarea 3D nu a putut fi randata momentan.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void render3D();

    return () => {
      cancelled = true;
      if (mount) {
        mount.innerHTML = "";
      }
    };
  }, [molBlock, renderStyle]);

  return (
    <div className="space-y-3">
      <div className={`relative w-full overflow-hidden rounded-[24px] border border-white/5 bg-slate-950/90 ${heightClassName}`}>
        <div className="pointer-events-none absolute inset-x-10 top-0 h-24 rounded-full bg-forge-cyan/12 blur-3xl" />
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(36,214,234,0.1),transparent_52%),radial-gradient(circle_at_bottom,rgba(115,166,255,0.08),transparent_48%)]" />
        <div ref={viewerRef} className="relative z-[1] h-full w-full" />

        {!molBlock ? (
          <div className="absolute inset-0 z-[2] flex items-center justify-center p-6 text-center text-sm leading-7 text-slate-400">
            {emptyMessage}
          </div>
        ) : null}

        {loading ? (
          <div className="absolute inset-0 z-[3] flex items-center justify-center bg-slate-950/56 backdrop-blur-[2px]">
            <div className="flex items-center gap-3 rounded-full border border-white/10 bg-slate-950/90 px-4 py-2 text-xs uppercase tracking-[0.18em] text-cyan-50">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-forge-cyan/25 border-t-forge-cyan" />
              Randare 3D
            </div>
          </div>
        ) : null}

        {error ? (
          <div className="absolute inset-x-5 bottom-5 z-[3] rounded-2xl border border-rose-400/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            {error}
          </div>
        ) : null}
      </div>

      {caption ? <p className="text-xs leading-6 text-slate-400">{caption}</p> : null}
    </div>
  );
});
