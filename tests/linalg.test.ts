import { describe, it, expect } from "vitest";
import * as L from "@/lib/math/linalg";

const close = (a: number, b: number, tol = 1e-8) => expect(Math.abs(a - b)).toBeLessThan(tol);
const closeM = (A: number[][], B: number[][], tol = 1e-8) => A.forEach((r, i) => r.forEach((x, j) => close(x, B[i][j], tol)));

describe("vectors", () => {
  it("dot, norms, cross", () => {
    expect(L.dot([1, 2, 3], [4, 5, 6])).toBe(32);
    close(L.norm2([3, 4]), 5);
    expect(L.norm1([3, -4])).toBe(7);
    expect(L.normInf([3, -4])).toBe(4);
    expect(L.cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
  });
  it("angle and projection", () => {
    close(L.angleBetween([1, 0], [0, 2])!, Math.PI / 2);
    expect(L.angleBetween([0, 0], [1, 1])).toBeNull();
    expect(L.project([3, 4], [1, 0])).toEqual([3, 0]);
    close(L.cosineSimilarity([1, 1], [2, 2])!, 1);
  });
  it("distances", () => {
    expect(L.manhattan([0, 0], [3, 4])).toBe(7);
    close(L.euclidean([0, 0], [3, 4]), 5);
    expect(L.chebyshev([0, 0], [3, 4])).toBe(4);
  });
});

describe("matrices", () => {
  it("multiplication", () => {
    expect(L.matMul([[1, 2], [3, 4]], [[5, 6], [7, 8]])).toEqual([[19, 22], [43, 50]]);
    expect(() => L.matMul([[1, 2]], [[1, 2]])).toThrow();
    expect(L.matVec([[1, 2], [3, 4]], [1, 1])).toEqual([3, 7]);
  });
  it("transpose, trace, hadamard, power", () => {
    expect(L.transpose([[1, 2, 3], [4, 5, 6]])).toEqual([[1, 4], [2, 5], [3, 6]]);
    expect(L.trace([[1, 2], [3, 4]])).toBe(5);
    expect(L.hadamard([[1, 2], [3, 4]], [[2, 2], [2, 2]])).toEqual([[2, 4], [6, 8]]);
    expect(L.matPow([[1, 1], [1, 0]], 10)).toEqual([[89, 55], [55, 34]]);
  });
  it("determinants", () => {
    expect(L.det2([[3, 8], [4, 6]])).toBe(-14);
    expect(L.det3([[6, 1, 1], [4, -2, 5], [2, 8, 7]])).toBe(-306);
    close(L.det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]), -306);
    close(L.det([[1, 2], [2, 4]]), 0);
    close(L.det([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]]), 120);
  });
  it("inverse and solve", () => {
    closeM(L.inverse([[4, 7], [2, 6]])!, [[0.6, -0.7], [-0.2, 0.4]]);
    expect(L.inverse([[1, 2], [2, 4]])).toBeNull();
    const A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]];
    const x = L.solve(A, [8, -11, -3])!;
    [2, 3, -1].forEach((v, i) => close(x[i], v));
    closeM(L.matMul(A, L.inverse(A)!), L.identity(3));
    closeM(L.adjugate([[1, 2], [3, 4]]), [[4, -2], [-3, 1]]);
  });
  it("rref, rank, null space", () => {
    const { result } = L.rrefSteps([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    closeM(result, [[1, 0, -1], [0, 1, 2], [0, 0, 0]]);
    expect(L.rank([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).toBe(2);
    const ns = L.nullSpace([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expect(ns.length).toBe(1);
    const Av = L.matVec([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ns[0]);
    Av.forEach((v) => close(v, 0));
  });
  it("LU", () => {
    const A = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]];
    const { L: Lm, U, P } = L.lu(A)!;
    closeM(L.matMul(Lm, U), L.matMul(P, A));
  });
});

describe("eigen & decompositions", () => {
  it("2x2 eigen", () => {
    const e = L.eigen2([[2, 1], [1, 2]]);
    close(e.values[0], 3);
    close(e.values[1], 1);
    const v = e.vectors![0];
    const Av = L.matVec([[2, 1], [1, 2]], v);
    close(Av[0], 3 * v[0]);
    close(Av[1], 3 * v[1]);
    const rot = L.eigen2([[0, -1], [1, 0]]);
    expect(rot.real).toBe(false);
    close(rot.imag, 1);
  });
  it("symmetric eigen (Jacobi)", () => {
    const S = [[4, 1, 2], [1, 2, 0], [2, 0, 3]];
    const { values, vectors } = L.eigSym(S);
    for (let k = 0; k < 3; k++) {
      const v = vectors.map((r) => r[k]);
      const Sv = L.matVec(S, v);
      Sv.forEach((x, i) => close(x, values[k] * v[i], 1e-7));
    }
    close(values.reduce((a, b) => a + b, 0), L.trace(S));
  });
  it("SVD reconstructs A", () => {
    const A = [[3, 2, 2], [2, 3, -2]];
    const { U, S, V } = L.svd(A);
    close(S[0], 5, 1e-7);
    close(S[1], 3, 1e-7);
    const Sig = [[S[0], 0], [0, S[1]]];
    const Vk = V.map((r) => r.slice(0, 2));
    closeM(L.matMul(L.matMul(U, Sig), L.transpose(Vk)), A, 1e-7);
  });
  it("QR, pinv, condition number", () => {
    const A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]];
    const { Q, R } = L.qr(A);
    closeM(L.matMul(Q, R), A, 1e-8);
    expect(L.isOrthogonal(Q, 1e-8)).toBe(true);
    const B = [[1, 2], [3, 4], [5, 6]];
    const Bp = L.pinv(B);
    closeM(L.matMul(L.matMul(B, Bp), B), B, 1e-7);
    close(L.conditionNumber([[1, 0], [0, 10]]), 10, 1e-7);
  });
  it("covariance", () => {
    const C = L.covarianceMatrix([[1, 2], [2, 4], [3, 6]]);
    closeM(C, [[1, 2], [2, 4]]);
  });
});
