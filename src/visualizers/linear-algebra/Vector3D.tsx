"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { MatrixInput, Presets, Toggle } from "@/components/controls";
import { useThemeRGB } from "@/components/visualization/Plot";
import { cross, add, norm2, det3, dot, angleBetween, type Matrix } from "@/lib/math/linalg";
import { fmt, tex } from "@/lib/utilities/format";

type V3 = [number, number, number];
// Maths uses z-up; three.js uses y-up. Map (x, y, z) → (x, z, −y), which keeps the frame right-handed.
const toThree = (v: number[]): [number, number, number] => [v[0], v[2], -v[1]];
const rgb = (c: [number, number, number]) => new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);

/**
 * Text label rendered as a camera-facing sprite with a canvas texture.
 * Replaces drei <Html>, which mounts a second React root per label and injects DOM nodes
 * into the Canvas wrapper — the source of "synchronously unmount a root" and removeChild errors.
 * Lives entirely inside the R3F scene graph; texture and material are disposed on unmount.
 */
function TextSprite({ text, position, color, size = 0.32 }: { text: string; position: [number, number, number]; color: THREE.Color; size?: number }) {
  const hex = color.getHexString();
  const { tex: texture, aspect } = useMemo(() => {
    const c = document.createElement("canvas");
    const g = c.getContext("2d")!;
    const font = "600 64px Inter, Segoe UI, sans-serif";
    g.font = font;
    const w = Math.ceil(g.measureText(text).width) + 16;
    c.width = w;
    c.height = 88;
    g.font = font;
    g.fillStyle = `#${hex}`;
    g.textBaseline = "middle";
    g.fillText(text, 8, 46);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    return { tex: t, aspect: w / 88 };
  }, [text, hex]);
  const material = useMemo(() => new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }), [texture]);
  useEffect(() => () => { texture.dispose(); material.dispose(); }, [texture, material]);
  return <sprite position={position} scale={[size * aspect, size, 1]} material={material} renderOrder={10} />;
}

function Arrow3({ to, from = [0, 0, 0], color, label, showLabel }: { to: number[]; from?: number[]; color: THREE.Color; label?: string; showLabel: boolean }) {
  const a = new THREE.Vector3(...toThree(from));
  const b = new THREE.Vector3(...toThree(to));
  const dir = b.clone().sub(a);
  const len = dir.length();
  const head = Math.min(0.28, len * 0.3);
  const q = useMemo(() => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize()), [dir.x, dir.y, dir.z]); // eslint-disable-line
  if (len < 1e-6) return null;
  const nd = dir.clone().normalize();
  const tip = b.clone().sub(nd.clone().multiplyScalar(head));
  const lp = b.clone().add(nd.clone().multiplyScalar(0.25));
  return (
    <group>
      <Line points={[a, tip]} color={color} lineWidth={2.5} />
      <mesh position={tip.clone().add(nd.clone().multiplyScalar(head / 2))} quaternion={q}>
        <coneGeometry args={[head * 0.38, head, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
      {label && showLabel && <TextSprite text={label} position={[lp.x, lp.y + 0.12, lp.z]} color={color} />}
    </group>
  );
}

function Axes({ c, showLabels, showGrid }: { c: THREE.Color; showLabels: boolean; showGrid: boolean }) {
  return (
    <group>
      {([[[-3, 0, 0], [3, 0, 0], "x"], [[0, -3, 0], [0, 3, 0], "y"], [[0, 0, -3], [0, 0, 3], "z"]] as const).map(([a, b, l]) => (
        <group key={l}>
          <Line points={[toThree(a as unknown as number[]), toThree(b as unknown as number[])]} color={c} lineWidth={1} transparent opacity={0.5} />
          {showLabels && <TextSprite text={l} position={toThree((b as unknown as number[]).map((v) => v * 1.08))} color={c} size={0.26} />}
        </group>
      ))}
      {showGrid && <gridHelper args={[6, 12, c, c]} position={[0, 0, 0]} material-opacity={0.15} material-transparent />}
    </group>
  );
}

/** Resets the orbit camera when `token` changes. */
function CameraReset({ token }: { token: number }) {
  const { camera, invalidate } = useThree();
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    camera.position.set(4.2, 3.4, 4.6);
    camera.lookAt(0, 0, 0);
    invalidate();
  }, [token, camera, invalidate]);
  return null;
}

function Parallelogram({ a, b, color, origin = [0, 0, 0] }: { a: number[]; b: number[]; color: THREE.Color; origin?: number[] }) {
  const geom = useMemo(() => {
    const o = origin;
    const pts = [o, add(o, a), add(add(o, a), b), add(o, b)].map((p) => new THREE.Vector3(...toThree(p)));
    const g = new THREE.BufferGeometry().setFromPoints([pts[0], pts[1], pts[2], pts[0], pts[2], pts[3]]);
    g.computeVertexNormals();
    return g;
  }, [a, b, origin]);
  return (
    <mesh geometry={geom}>
      <meshBasicMaterial color={color} transparent opacity={0.22} side={THREE.DoubleSide} depthWrite={false} />
    </mesh>
  );
}

function Parallelepiped({ M, color }: { M: Matrix; color: THREE.Color }) {
  const geom = useMemo(() => {
    const c = [0, 1, 2].map((j) => M.map((r) => r[j]));
    const g = new THREE.BoxGeometry(1, 1, 1);
    g.translate(0.5, 0.5, 0.5);
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const u = pos.getX(i), v = pos.getY(i), w = pos.getZ(i);
      const p = [0, 1, 2].map((k) => u * c[0][k] + v * c[1][k] + w * c[2][k]);
      const t = toThree(p);
      pos.setXYZ(i, t[0], t[1], t[2]);
    }
    pos.needsUpdate = true;
    return g;
  }, [M]);
  const edges = useMemo(() => new THREE.EdgesGeometry(geom), [geom]);
  return (
    <group>
      <mesh geometry={geom}>
        <meshBasicMaterial color={color} transparent opacity={0.18} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={color} transparent opacity={0.8} />
      </lineSegments>
    </group>
  );
}

export default function Vector3D({ mode = "add" }: { mode?: "add" | "cross" | "det3" }) {
  const th = useThemeRGB();
  const [U, setU] = useState<Matrix>([[2, 0.5, 0.5]]);
  const [Vv, setVv] = useState<Matrix>([[0.5, 2, 1]]);
  const [M, setM] = useState<Matrix>([[1.5, 0.3, 0], [0.2, 1.2, 0.4], [0, 0.3, 1]]);
  const [showLabels, setShowLabels] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [camToken, setCamToken] = useState(0);
  const u = U[0] as V3, v = Vv[0] as V3;
  const s = add(u, v);
  const cx = cross(u, v);
  const D = det3(M);
  const colors = { blue: rgb(th.blue), pink: rgb(th.pink), green: rgb(th.green), amber: rgb(th.amber), axis: rgb(th.muted) };
  const ang = angleBetween(u, v);

  const readouts =
    mode === "det3"
      ? [{ label: "\\det A", tex: true, value: fmt(D), color: Math.abs(D) < 1e-9 ? "rgb(var(--red))" : undefined }, { label: "\\text{volume scale}", tex: true, value: `×${fmt(Math.abs(D))}` }, { label: "\\text{orientation}", tex: true, value: Math.abs(D) < 1e-9 ? "collapsed" : D > 0 ? "preserved" : "flipped" }]
      : mode === "cross"
        ? [{ label: "\\mathbf u\\times\\mathbf v", tex: true, value: `(${cx.map((x) => fmt(x)).join(", ")})` }, { label: "\\|\\mathbf u\\times\\mathbf v\\|", tex: true, value: fmt(norm2(cx)) }, { label: "(\\mathbf u\\times\\mathbf v)\\cdot\\mathbf u", tex: true, value: fmt(dot(cx, u), 9) }, { label: "\\|\\mathbf u\\|\\|\\mathbf v\\|\\sin\\theta", tex: true, value: ang === null ? "—" : fmt(norm2(u) * norm2(v) * Math.sin(ang)) }]
        : [{ label: "\\mathbf u+\\mathbf v", tex: true, value: `(${s.map((x) => fmt(x)).join(", ")})` }, { label: "\\|\\mathbf u+\\mathbf v\\|", tex: true, value: fmt(norm2(s)) }, { label: "\\|\\mathbf u\\|", tex: true, value: fmt(norm2(u)) }, { label: "\\|\\mathbf v\\|", tex: true, value: fmt(norm2(v)) }];

  const steps =
    mode === "det3"
      ? [{ tex: `\\det A=${tex(M[0][0], 2)}(${tex(M[1][1], 2)}\\cdot${tex(M[2][2], 2)}-${tex(M[1][2], 2)}\\cdot${tex(M[2][1], 2)})-${tex(M[0][1], 2)}(${tex(M[1][0], 2)}\\cdot${tex(M[2][2], 2)}-${tex(M[1][2], 2)}\\cdot${tex(M[2][0], 2)})+${tex(M[0][2], 2)}(${tex(M[1][0], 2)}\\cdot${tex(M[2][1], 2)}-${tex(M[1][1], 2)}\\cdot${tex(M[2][0], 2)})=${tex(D)}` }]
      : mode === "cross"
        ? [{ tex: `\\mathbf u\\times\\mathbf v=(${tex(u[1], 2)}\\cdot${tex(v[2], 2)}-${tex(u[2], 2)}\\cdot${tex(v[1], 2)},\\;${tex(u[2], 2)}\\cdot${tex(v[0], 2)}-${tex(u[0], 2)}\\cdot${tex(v[2], 2)},\\;${tex(u[0], 2)}\\cdot${tex(v[1], 2)}-${tex(u[1], 2)}\\cdot${tex(v[0], 2)})=(${cx.map((x) => tex(x)).join(",")})` }]
        : [{ tex: `(${u.map((x) => tex(x, 2)).join(",")})+(${v.map((x) => tex(x, 2)).join(",")})=(${s.map((x) => tex(x, 2)).join(",")})` }];

  return (
    <LabShell
      viz={
        <div className="h-[420px] w-full overflow-hidden rounded-md sm:h-[480px]" role="img" aria-label={`3D vector scene: ${mode}. Values are listed below.`}>
          <Canvas camera={{ position: [4.2, 3.4, 4.6], fov: 45 }} frameloop="demand" dpr={[1, 2]}>
            {showAxes && <Axes c={colors.axis} showLabels={showLabels} showGrid={showGrid} />}
            <CameraReset token={camToken} />
            {mode === "add" && (
              <>
                <Line points={[toThree(u), toThree(s)]} color={colors.pink} lineWidth={1} dashed dashSize={0.15} gapSize={0.1} />
                <Line points={[toThree(v), toThree(s)]} color={colors.blue} lineWidth={1} dashed dashSize={0.15} gapSize={0.1} />
                <Parallelogram a={u} b={v} color={colors.green} />
                <Arrow3 to={u} color={colors.blue} label="u" showLabel={showLabels} />
                <Arrow3 to={v} color={colors.pink} label="v" showLabel={showLabels} />
                <Arrow3 to={s} color={colors.green} label="u+v" showLabel={showLabels} />
              </>
            )}
            {mode === "cross" && (
              <>
                <Parallelogram a={u} b={v} color={colors.amber} />
                <Arrow3 to={u} color={colors.blue} label="u" showLabel={showLabels} />
                <Arrow3 to={v} color={colors.green} label="v" showLabel={showLabels} />
                <Arrow3 to={cx} color={colors.pink} label="u×v" showLabel={showLabels} />
              </>
            )}
            {mode === "det3" && (
              <>
                <Parallelepiped M={[[1, 0, 0], [0, 1, 0], [0, 0, 1]]} color={colors.axis} />
                <Parallelepiped M={M} color={Math.abs(D) < 1e-9 ? rgb(th.red) : colors.pink} />
                <Arrow3 to={M.map((r) => r[0])} color={colors.blue} label="Ae₁" showLabel={showLabels} />
                <Arrow3 to={M.map((r) => r[1])} color={colors.green} label="Ae₂" showLabel={showLabels} />
                <Arrow3 to={M.map((r) => r[2])} color={colors.amber} label="Ae₃" showLabel={showLabels} />
              </>
            )}
            <OrbitControls key={camToken} enableDamping={false} enablePan minDistance={3} maxDistance={14} />
          </Canvas>
        </div>
      }
      controls={
        <>
        {mode === "det3" ? (
          <>
            <MatrixInput label="A (3×3)" value={M} onChange={setM} step={0.1} />
            <Presets presets={[{ label: "Identity", value: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }, { label: "Shear", value: [[1, 0.8, 0], [0, 1, 0.5], [0, 0, 1]] }, { label: "Singular", value: [[1, 2, 1], [0, 1, 1], [1, 3, 2]] }, { label: "Reflection", value: [[1, 0, 0], [0, 1, 0], [0, 0, -1]] }]} onPick={(m) => setM(m)} />
          </>
        ) : (
          <>
            <MatrixInput label="u" value={U} onChange={setU} step={0.1} />
            <MatrixInput label="v" value={Vv} onChange={setVv} step={0.1} />
            <Presets presets={[{ label: "e₁, e₂", value: [[1, 0, 0], [0, 1, 0]] }, { label: "Parallel", value: [[1, 1, 0], [2, 2, 0]] }, { label: "General", value: [[2, 0.5, 0.5], [0.5, 2, 1]] }]} onPick={([a, b]) => { setU([a]); setVv([b]); }} />
          </>
        )}
        <div className="space-y-2 border-t border-line pt-4">
          <Toggle label="Labels" checked={showLabels} onChange={setShowLabels} />
          <Toggle label="Axes" checked={showAxes} onChange={setShowAxes} />
          <Toggle label="Ground grid" checked={showGrid} onChange={setShowGrid} />
          <button className="btn w-full" onClick={() => setCamToken((t) => t + 1)}>Reset view</button>
        </div>
        </>
      }
      readouts={<Readouts items={readouts} />}
      calc={<CalcSteps steps={steps} />}
      note="Drag to orbit, right-drag to pan, scroll to zoom. Axes use the mathematical right-handed convention with z pointing up."
    />
  );
}
