import type { SectionDef, TopicDef } from "@/types/curriculum";

const r = String.raw;

/* Topics added after auditing the mathematical curriculum for gaps. Each one fills a hole that the
   existing lessons referred to (null space, projection matrices, the spectral theorem, the pseudoinverse,
   gradients of matrix expressions, joint distributions, convex sets) without teaching it directly. */

/* ------------------------------------------------------------------ Linear algebra */
const fourSubspaces: TopicDef = {
  slug: "four-fundamental-subspaces",
  title: "Null Space and the Four Fundamental Subspaces",
  summary: "Column space, row space and the two null spaces, tied together by rank.",
  difficulty: "intermediate",
  keywords: ["null space", "kernel", "column space", "row space", "left null space", "rank nullity", "fundamental subspaces", "orthogonal complement"],
  prerequisites: ["linear-algebra/rank", "linear-algebra/vector-spaces", "linear-algebra/gaussian-elimination"],
  related: ["linear-algebra/pseudoinverse", "linear-algebra/projection-matrices", "linear-algebra/linear-systems", "linear-algebra/svd"],
  content: {
    intuition: "A matrix is a machine that takes input vectors to output vectors. Some input directions are crushed to zero — the machine cannot see them. The outputs it can produce do not usually fill the whole output space either. Four subspaces record exactly what is seen, what is lost, what can be reached and what cannot.",
    definition: "For an m × n matrix A, the column space C(A) is the set of all outputs Ax; the null space N(A) is the set of inputs with Ax = 0; the row space C(Aᵀ) is the span of the rows; and the left null space N(Aᵀ) is the set of y with Aᵀy = 0. Their dimensions are r, n − r, r and m − r, where r is the rank. Row space and null space are orthogonal complements in ℝⁿ; column space and left null space are orthogonal complements in ℝᵐ.",
    why: "They answer the basic questions about a linear system: Ax = b is solvable exactly when b is in the column space, and its solution is unique exactly when the null space contains only zero. In ML they explain unidentifiable weights, redundant features and what a linear layer discards.",
    usedIn: ["Solvability and uniqueness of linear systems", "Identifiability of regression coefficients", "Understanding rank-deficient weight matrices and low-rank adaptation"],
    visual: { id: "subspaces", caption: "Edit a 2 × 2 matrix and drag the input x. Left: x split into its row-space and null-space parts. Right: the output Ax, which always lies in the column space and depends only on the row-space part." },
    formulas: [
      { label: "The four subspaces", tex: r`C(A)=\{A\mathbf x\}\subseteq\mathbb R^m,\quad N(A)=\{\mathbf x:A\mathbf x=\mathbf 0\}\subseteq\mathbb R^n,\quad C(A^\top)\subseteq\mathbb R^n,\quad N(A^\top)\subseteq\mathbb R^m`, explanation: "Two live in the input space and two in the output space.", variables: [{ symbol: "A", meaning: "an m × n matrix" }, { symbol: "r", meaning: "rank of A" }] },
      { label: "Rank–nullity theorem", tex: r`\dim C(A^\top)+\dim N(A)=r+(n-r)=n`, explanation: "Every input dimension is either seen by A (row space) or destroyed by it (null space)." },
      { label: "Orthogonality", tex: r`N(A)=C(A^\top)^{\perp},\qquad N(A^\top)=C(A)^{\perp}`, explanation: "Ax = 0 says x is orthogonal to every row of A." },
      { label: "Decomposition of an input", tex: r`\mathbf x=\mathbf x_{\text{row}}+\mathbf x_{\text{null}},\qquad A\mathbf x=A\mathbf x_{\text{row}}`, explanation: "Only the row-space component affects the output." },
      { label: "Solutions of a linear system", tex: r`A\mathbf x=\mathbf b\ \text{solvable}\iff\mathbf b\in C(A);\qquad\text{all solutions}=\mathbf x_p+N(A)`, explanation: "One particular solution plus anything from the null space." },
    ],
    derivation: {
      title: "the null space is orthogonal to the row space",
      steps: [
        { tex: r`A\mathbf x=\mathbf 0\iff\mathbf r_i^\top\mathbf x=0\ \ \text{for every row }\mathbf r_i\text{ of }A`, note: "Each entry of Ax is the dot product of a row with x." },
        { tex: r`\mathbf v=\sum_ic_i\mathbf r_i\in C(A^\top)\ \Rightarrow\ \mathbf v^\top\mathbf x=\sum_ic_i\,\mathbf r_i^\top\mathbf x=0`, note: "So x is orthogonal to every combination of rows." },
        { tex: r`\dim C(A^\top)+\dim N(A)=n\ \Rightarrow\ N(A)=C(A^\top)^\perp`, note: "The dimensions add up to n, so the null space is the entire orthogonal complement, not just part of it." },
      ],
      conclusion: "ℝⁿ splits into two perpendicular pieces, one that A maps faithfully onto its column space and one that it sends to zero. Applying the same argument to Aᵀ gives the split of ℝᵐ.",
    },
    specialCases: ["Square invertible A: N(A) = {0} and C(A) = ℝⁿ.", "A wide matrix (n > m) always has a non-trivial null space: more unknowns than equations."],
    limitations: ["Numerically, rank is not clear-cut: tiny singular values make a matrix 'almost' rank-deficient. The SVD gives orthonormal bases for all four subspaces and a principled threshold."],
    worked: {
      title: "a rank-1 matrix",
      steps: [
        { tex: r`A=\begin{bmatrix}1&2\\2&4\end{bmatrix},\qquad\operatorname{rref}(A)=\begin{bmatrix}1&2\\0&0\end{bmatrix}\ \Rightarrow\ r=1`, note: "one pivot" },
        { tex: r`N(A)=\operatorname{span}\{(-2,1)\},\qquad C(A^\top)=\operatorname{span}\{(1,2)\}`, note: "free variable x₂ = 1 gives x₁ = −2; the row space is spanned by the pivot row" },
        { tex: r`(-2,1)\cdot(1,2)=0`, note: "perpendicular, as the theorem says" },
        { tex: r`C(A)=\operatorname{span}\{(1,2)\},\qquad N(A^\top)=\operatorname{span}\{(-2,1)\}`, note: "A is symmetric here, so the output-space pair coincides with the input-space pair" },
        { tex: r`\mathbf x=(2,0.5)=(0.6,1.2)+(1.4,-0.7)\ \Rightarrow\ A\mathbf x=A(0.6,1.2)^\top=(3,6)^\top`, note: "row-space part plus null-space part; only the first matters" },
      ],
      result: r`\dim C(A^\top)+\dim N(A)=1+1=2`,
    },
    applications: [
      { title: "Unidentifiable regression weights", body: "If the feature matrix has a non-trivial null space, adding any null-space vector to the weights leaves every prediction unchanged: the data cannot determine the weights uniquely. Regularisation or the pseudoinverse picks one." },
      { title: "What a layer forgets", body: "A linear layer with fewer outputs than inputs has a null space of dimension at least n − m; information along those directions is irrecoverably lost to later layers." },
    ],
  },
};

const projectionMatrices: TopicDef = {
  slug: "projection-matrices",
  title: "Orthogonal Projection onto a Subspace",
  summary: "The matrix P = A(AᵀA)⁻¹Aᵀ that drops any vector onto the column space of A.",
  difficulty: "intermediate",
  keywords: ["projection matrix", "orthogonal projection", "subspace projection", "hat matrix", "idempotent", "closest point", "A(A^T A)^-1 A^T"],
  prerequisites: ["linear-algebra/projection", "linear-algebra/matrix-inverse", "linear-algebra/span-and-basis"],
  related: ["linear-algebra/least-squares", "linear-algebra/four-fundamental-subspaces", "linear-algebra/gram-schmidt", "linear-algebra/mlt-regression-views"],
  content: {
    intuition: "Hold a stick above a table under a lamp directly overhead. Its shadow is the closest thing to the stick that lies in the table's plane, and the line from the stick's tip to the shadow's tip is perpendicular to the table. Projection onto a subspace is that shadow, in any number of dimensions.",
    definition: "The orthogonal projection of b onto the subspace spanned by the independent columns of A is p = Pb with P = A(AᵀA)⁻¹Aᵀ. It is the point of the subspace closest to b, and the error e = b − p is orthogonal to the subspace. A matrix is an orthogonal projection exactly when it is symmetric and idempotent: Pᵀ = P and P² = P.",
    why: "It generalises projecting onto a single vector, and it is the geometric content of least squares: fitted values are the projection of the labels onto the space spanned by the features. PCA reconstruction and Gram–Schmidt are projections too.",
    usedIn: ["Least-squares regression (the 'hat' matrix)", "PCA reconstruction onto the top components", "Removing a known direction or subspace from data"],
    visual: { id: "subspace-projection", caption: "A plane in ℝ³ spanned by the two columns of A, a vector b, its projection p and the error e. Edit the numbers and rotate the view; the readouts verify that Aᵀe = 0 and P² = P." },
    formulas: [
      { label: "Projection onto a line", tex: r`P=\frac{\mathbf a\mathbf a^\top}{\mathbf a^\top\mathbf a},\qquad\mathbf p=P\mathbf b=\frac{\mathbf a^\top\mathbf b}{\mathbf a^\top\mathbf a}\,\mathbf a`, explanation: "The one-column case, already met as vector projection.", variables: [{ symbol: r`\mathbf a`, meaning: "vector spanning the line" }, { symbol: r`\mathbf b`, meaning: "vector being projected" }] },
      { label: "Projection onto a subspace", tex: r`P=A(A^\top A)^{-1}A^\top,\qquad\mathbf p=P\mathbf b=A\hat{\mathbf x},\quad\hat{\mathbf x}=(A^\top A)^{-1}A^\top\mathbf b`, explanation: "x̂ holds the coordinates of the projection in the basis given by the columns of A.", variables: [{ symbol: "A", meaning: "m × k matrix with linearly independent columns" }, { symbol: "P", meaning: "m × m projection matrix" }] },
      { label: "Defining properties", tex: r`P^2=P,\qquad P^\top=P`, explanation: "Projecting twice changes nothing; symmetry makes the projection orthogonal rather than oblique." },
      { label: "Orthonormal basis", tex: r`Q^\top Q=I\ \Longrightarrow\ P=QQ^\top`, explanation: "With orthonormal columns the inverse disappears — the reason Gram–Schmidt and QR are useful." },
      { label: "Complementary projection", tex: r`I-P\ \text{projects onto}\ C(A)^\perp,\qquad\mathbf b=P\mathbf b+(I-P)\mathbf b`, explanation: "Every vector splits into a part in the subspace and a part perpendicular to it." },
    ],
    derivation: {
      title: "deriving P from the orthogonality of the error",
      setup: "Seek the point p = Ax̂ in the column space whose error e = b − Ax̂ is perpendicular to every column of A.",
      steps: [
        { tex: r`A^\top(\mathbf b-A\hat{\mathbf x})=\mathbf 0`, note: "Orthogonality to each column, written as one matrix equation." },
        { tex: r`A^\top A\,\hat{\mathbf x}=A^\top\mathbf b\ \Rightarrow\ \hat{\mathbf x}=(A^\top A)^{-1}A^\top\mathbf b`, note: "The normal equations. AᵀA is invertible because the columns of A are independent." },
        { tex: r`\mathbf p=A\hat{\mathbf x}=A(A^\top A)^{-1}A^\top\mathbf b=P\mathbf b`, note: "Read off P." },
        { tex: r`P^2=A(A^\top A)^{-1}\underbrace{A^\top A(A^\top A)^{-1}}_{I}A^\top=P`, note: "Idempotence." },
        { tex: r`\|\mathbf b-\mathbf v\|^2=\|\mathbf b-\mathbf p\|^2+\|\mathbf p-\mathbf v\|^2\ \ge\ \|\mathbf b-\mathbf p\|^2\quad\text{for any }\mathbf v\in C(A)`, note: "Pythagoras, because b − p is perpendicular to p − v: p is the closest point." },
      ],
      conclusion: "Closest point and perpendicular error are the same condition, and it is exactly the normal equations of least squares.",
    },
    assumptions: ["The columns of A are linearly independent, so that AᵀA is invertible. Otherwise use an orthonormal basis of the column space or the pseudoinverse: P = AA⁺."],
    specialCases: ["If b is already in the subspace, Pb = b.", "If b is orthogonal to the subspace, Pb = 0."],
    limitations: ["Forming (AᵀA)⁻¹ explicitly squares the condition number; numerically the projection is computed through QR or the SVD."],
    worked: {
      title: "projecting b = (1, 2, 3) onto a plane",
      steps: [
        { tex: r`A=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}\ (\text{the }xy\text{-plane}),\qquad A^\top A=I`, note: "orthonormal columns" },
        { tex: r`P=AA^\top=\begin{bmatrix}1&0&0\\0&1&0\\0&0&0\end{bmatrix},\qquad\mathbf p=P\mathbf b=(1,2,0)`, note: "the projection simply drops the z-coordinate" },
        { tex: r`\mathbf e=\mathbf b-\mathbf p=(0,0,3),\qquad A^\top\mathbf e=(0,0)`, note: "the error is perpendicular to the plane" },
        { tex: r`\text{line through }\mathbf a=(1,1,1):\quad\mathbf p=\frac{6}{3}(1,1,1)=(2,2,2),\quad\mathbf e=(-1,0,1),\quad\mathbf a^\top\mathbf e=0`, note: "the one-dimensional case for comparison" },
      ],
      result: r`\mathbf p=(1,2,0),\qquad\mathbf e=(0,0,3)`,
    },
    applications: [
      { title: "The hat matrix", body: "In regression the fitted values are ŷ = Hy with H the projection onto the feature space. Its diagonal entries (leverages) measure how strongly each observation influences its own fit, and give leave-one-out errors without refitting." },
      { title: "Removing a concept direction", body: "Multiplying embeddings by I − uuᵀ projects out the direction u; this is used to remove a nuisance direction (for example a bias direction) from learned representations." },
    ],
  },
};

const spectralTheorem: TopicDef = {
  slug: "spectral-theorem",
  title: "The Spectral Theorem for Symmetric Matrices",
  summary: "Real eigenvalues, orthonormal eigenvectors, and a sum of rank-one projections.",
  difficulty: "advanced",
  keywords: ["spectral theorem", "symmetric matrix", "orthogonal diagonalization", "real eigenvalues", "rank one decomposition", "Q Lambda Q^T", "eigendecomposition applications"],
  prerequisites: ["linear-algebra/eigenvalues-eigenvectors", "linear-algebra/orthogonal-matrices", "linear-algebra/diagonalization"],
  related: ["linear-algebra/positive-definite", "linear-algebra/svd", "linear-algebra/covariance-matrix", "linear-algebra/mlt-pca-principal-components"],
  content: {
    intuition: "A general matrix can rotate, shear and stretch in tangled ways. A symmetric matrix is far tamer: there is always a set of mutually perpendicular axes along which it does nothing but stretch or flip. In those axes the matrix is a list of scale factors.",
    definition: "Spectral theorem: every real symmetric n × n matrix A has n real eigenvalues λ₁, …, λₙ and an orthonormal basis of eigenvectors q₁, …, qₙ. Equivalently A = QΛQᵀ with Q orthogonal and Λ diagonal, or A = Σᵢ λᵢ qᵢqᵢᵀ — a weighted sum of projections onto perpendicular lines.",
    why: "The matrices that matter most in ML are symmetric: covariance matrices, kernel (Gram) matrices, Hessians, graph Laplacians. The theorem guarantees they can always be analysed in an orthonormal eigenbasis, which is what makes PCA, spectral clustering and second-order optimisation work.",
    usedIn: ["PCA (eigenvectors of the covariance matrix)", "Curvature analysis via Hessian eigenvalues", "Spectral clustering and graph methods", "Matrix functions such as A^{1/2} and A⁻¹ for whitening"],
    visual: { id: "spectral-decomp", caption: "A symmetric 2 × 2 matrix set by three sliders. The eigen-directions stay perpendicular whatever you choose; drag x to see Ax assembled from its two eigen-components. The readouts check trace and determinant against the eigenvalues." },
    formulas: [
      { label: "Spectral decomposition", tex: r`A=A^\top\ \Longrightarrow\ A=Q\Lambda Q^\top,\qquad Q^\top Q=I,\quad\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n),\ \lambda_i\in\mathbb R`, explanation: "Orthogonal diagonalisation: change to the eigenbasis, scale, change back.", variables: [{ symbol: "Q", meaning: "orthogonal matrix whose columns are the eigenvectors qᵢ" }, { symbol: r`\Lambda`, meaning: "diagonal matrix of eigenvalues" }] },
      { label: "Rank-one form", tex: r`A=\sum_{i=1}^{n}\lambda_i\,\mathbf q_i\mathbf q_i^\top`, explanation: "Each qᵢqᵢᵀ is the projection onto the line spanned by qᵢ." },
      { label: "Action on a vector", tex: r`A\mathbf x=\sum_{i=1}^{n}\lambda_i\,(\mathbf q_i^\top\mathbf x)\,\mathbf q_i`, explanation: "Split x along the eigenvectors, scale each component by its eigenvalue, add." },
      { label: "Functions of a matrix", tex: r`A^k=Q\Lambda^kQ^\top,\qquad A^{-1}=Q\Lambda^{-1}Q^\top,\qquad A^{1/2}=Q\Lambda^{1/2}Q^\top`, explanation: "Any function is applied to the eigenvalues alone (the inverse needs λᵢ ≠ 0, the square root λᵢ ≥ 0)." },
      { label: "Rayleigh quotient", tex: r`\lambda_{\min}\le\frac{\mathbf x^\top A\mathbf x}{\mathbf x^\top\mathbf x}\le\lambda_{\max}`, explanation: "The quadratic form on unit vectors ranges exactly between the extreme eigenvalues — the fact behind PCA's variance maximisation." },
    ],
    derivation: {
      title: "real eigenvalues and orthogonal eigenvectors",
      steps: [
        { tex: r`A\mathbf v=\lambda\mathbf v\ \Rightarrow\ \bar{\mathbf v}^\top A\mathbf v=\lambda\,\bar{\mathbf v}^\top\mathbf v`, note: "Allow complex eigenpairs for the moment; the bar denotes complex conjugation." },
        { tex: r`\overline{\bar{\mathbf v}^\top A\mathbf v}=\mathbf v^\top A\bar{\mathbf v}=\bar{\mathbf v}^\top A^\top\mathbf v=\bar{\mathbf v}^\top A\mathbf v`, note: "A is real and symmetric, so this number equals its own conjugate: it is real." },
        { tex: r`\lambda=\frac{\bar{\mathbf v}^\top A\mathbf v}{\bar{\mathbf v}^\top\mathbf v}\in\mathbb R`, note: "A real number divided by a positive real number." },
        { tex: r`A\mathbf u=\lambda\mathbf u,\ A\mathbf v=\mu\mathbf v:\quad\lambda\,\mathbf u^\top\mathbf v=(A\mathbf u)^\top\mathbf v=\mathbf u^\top A\mathbf v=\mu\,\mathbf u^\top\mathbf v`, note: "Move A from one side of the inner product to the other using symmetry." },
        { tex: r`(\lambda-\mu)\,\mathbf u^\top\mathbf v=0\ \Rightarrow\ \mathbf u\perp\mathbf v\ \text{ if }\lambda\neq\mu`, note: "Eigenvectors for different eigenvalues are orthogonal." },
      ],
      conclusion: "Within a repeated eigenvalue's eigenspace an orthonormal basis can be chosen (Gram–Schmidt), and symmetric matrices are never defective, so a full orthonormal eigenbasis always exists.",
    },
    assumptions: ["A is real and symmetric (or complex Hermitian). Non-symmetric matrices can have complex eigenvalues and non-orthogonal or missing eigenvectors."],
    specialCases: ["All λᵢ > 0: positive definite, and the unit circle maps to an ellipse with semi-axes λᵢ along qᵢ.", "Repeated eigenvalues: the eigenvectors are not unique, but the eigenspaces and the matrix QΛQᵀ are."],
    limitations: ["For non-symmetric or rectangular matrices the right generalisation is the singular value decomposition A = UΣVᵀ, built from the spectral decompositions of AᵀA and AAᵀ."],
    worked: {
      title: "decomposing [[2, 1], [1, 2]]",
      steps: [
        { tex: r`\det(A-\lambda I)=(2-\lambda)^2-1=0\ \Rightarrow\ \lambda_1=3,\ \lambda_2=1`, note: "both real" },
        { tex: r`\mathbf q_1=\tfrac{1}{\sqrt2}(1,1),\qquad\mathbf q_2=\tfrac{1}{\sqrt2}(1,-1),\qquad\mathbf q_1^\top\mathbf q_2=0`, note: "orthonormal eigenvectors" },
        { tex: r`3\,\mathbf q_1\mathbf q_1^\top+1\,\mathbf q_2\mathbf q_2^\top=\tfrac32\begin{bmatrix}1&1\\1&1\end{bmatrix}+\tfrac12\begin{bmatrix}1&-1\\-1&1\end{bmatrix}=\begin{bmatrix}2&1\\1&2\end{bmatrix}`, note: "the rank-one pieces rebuild A" },
        { tex: r`\mathbf x=(1,0):\ \ \mathbf q_1^\top\mathbf x=\mathbf q_2^\top\mathbf x=\tfrac{1}{\sqrt2}\ \Rightarrow\ A\mathbf x=3\cdot\tfrac{1}{\sqrt2}\mathbf q_1+1\cdot\tfrac{1}{\sqrt2}\mathbf q_2=(2,1)`, note: "matches the first column of A" },
        { tex: r`A^{-1}=\tfrac13\mathbf q_1\mathbf q_1^\top+1\,\mathbf q_2\mathbf q_2^\top=\tfrac13\begin{bmatrix}2&-1\\-1&2\end{bmatrix}`, note: "invert the eigenvalues, keep the eigenvectors" },
      ],
      result: r`A=3\,\mathbf q_1\mathbf q_1^\top+\mathbf q_2\mathbf q_2^\top`,
    },
    applications: [
      { title: "PCA", body: "The covariance matrix is symmetric, so its eigenvectors are orthonormal and its eigenvalues real and non-negative: the principal components and the variances along them." },
      { title: "Loss curvature", body: "The Hessian of a loss is symmetric. Its eigenvalues are the curvatures along perpendicular directions: all positive at a minimum, mixed signs at a saddle; the largest one limits the stable learning rate." },
    ],
  },
};

const pseudoinverse: TopicDef = {
  slug: "pseudoinverse",
  title: "The Moore–Penrose Pseudoinverse",
  summary: "The best substitute for an inverse when none exists.",
  difficulty: "advanced",
  keywords: ["pseudoinverse", "moore penrose", "minimum norm solution", "least squares solution", "A plus", "generalized inverse", "underdetermined", "overdetermined"],
  prerequisites: ["linear-algebra/svd", "linear-algebra/least-squares", "linear-algebra/four-fundamental-subspaces"],
  related: ["linear-algebra/projection-matrices", "linear-algebra/condition-number", "linear-algebra/matrix-inverse", "linear-algebra/mlt-linear-regression"],
  content: {
    intuition: "An inverse undoes a matrix, but a matrix that crushes some directions or cannot reach some outputs has nothing to undo them with. The pseudoinverse does the most sensible thing in both cases: for an unreachable target it aims at the nearest reachable one, and among the many inputs that hit it, it returns the shortest.",
    definition: "For any m × n matrix A with singular value decomposition A = UΣVᵀ, the Moore–Penrose pseudoinverse is A⁺ = VΣ⁺Uᵀ, where Σ⁺ inverts the non-zero singular values and leaves the zeros. The vector x⁺ = A⁺b minimises ‖Ax − b‖, and among all minimisers it has the smallest norm ‖x‖.",
    why: "It gives one formula that covers every linear system: square or rectangular, full rank or not. It is what a least-squares solver returns, and it explains which solution over-parameterised models converge to.",
    usedIn: ["Least-squares regression, including rank-deficient cases", "Minimum-norm interpolation in over-parameterised models", "Solving linear systems robustly in numerical libraries"],
    visual: { id: "pinv", caption: "Two situations. Fewer equations than unknowns: a whole line of exact solutions, of which A⁺b is the one nearest the origin. More equations than unknowns: three lines with no common point, and A⁺b where the squared distances to them are smallest." },
    formulas: [
      { label: "Definition through the SVD", tex: r`A=U\Sigma V^\top\ \Longrightarrow\ A^+=V\Sigma^+U^\top,\qquad\Sigma^+_{ii}=\begin{cases}1/\sigma_i&\sigma_i>0\\0&\sigma_i=0\end{cases}`, explanation: "Invert what can be inverted.", variables: [{ symbol: r`\sigma_i`, meaning: "singular values of A" }, { symbol: "A^+", meaning: "n × m pseudoinverse" }] },
      { label: "Full column rank (tall matrix)", tex: r`A^+=(A^\top A)^{-1}A^\top\qquad(A^+A=I)`, explanation: "A left inverse: gives the unique least-squares solution." },
      { label: "Full row rank (wide matrix)", tex: r`A^+=A^\top(AA^\top)^{-1}\qquad(AA^+=I)`, explanation: "A right inverse: gives the minimum-norm exact solution." },
      { label: "What A⁺b solves", tex: r`\mathbf x^+=A^+\mathbf b=\arg\min_{\mathbf x\,\in\,\arg\min\|A\mathbf x-\mathbf b\|}\|\mathbf x\|`, explanation: "Least squares first; shortest vector among the least-squares solutions second." },
      { label: "Projections", tex: r`AA^+=\text{projection onto }C(A),\qquad A^+A=\text{projection onto }C(A^\top)`, explanation: "The pseudoinverse inverts A between its row space and its column space and annihilates everything else." },
    ],
    derivation: {
      title: "why A⁺b is the minimum-norm least-squares solution",
      setup: "Change variables with the orthogonal matrices of the SVD: c = Uᵀb and z = Vᵀx. Orthogonal maps preserve lengths.",
      steps: [
        { tex: r`\|A\mathbf x-\mathbf b\|^2=\|U\Sigma V^\top\mathbf x-\mathbf b\|^2=\|\Sigma\mathbf z-\mathbf c\|^2`, note: "Multiply inside the norm by Uᵀ." },
        { tex: r`=\sum_{i:\sigma_i>0}(\sigma_iz_i-c_i)^2+\sum_{i:\sigma_i=0}c_i^2`, note: "Σ is diagonal, so the problem decouples coordinate by coordinate." },
        { tex: r`\text{minimised by }z_i=\frac{c_i}{\sigma_i}\ \ (\sigma_i>0);\qquad z_i\ \text{arbitrary}\ \ (\sigma_i=0)`, note: "The second sum cannot be reduced: it is the unavoidable residual." },
        { tex: r`\|\mathbf x\|=\|\mathbf z\|\ \text{is smallest when the free }z_i=0\ \Rightarrow\ \mathbf z=\Sigma^+\mathbf c`, note: "Among the minimisers, set the unconstrained coordinates to zero." },
        { tex: r`\mathbf x^+=V\mathbf z=V\Sigma^+U^\top\mathbf b=A^+\mathbf b`, note: "Transform back." },
      ],
      conclusion: "The SVD turns any linear system into independent scalar equations σᵢzᵢ = cᵢ; the pseudoinverse solves the solvable ones and sets the rest to zero.",
    },
    specialCases: ["If A is square and invertible, A⁺ = A⁻¹.", "For a non-zero column vector a, a⁺ = aᵀ/(aᵀa); for the zero matrix, 0⁺ = 0ᵀ."],
    limitations: ["Deciding which singular values count as zero requires a tolerance; values just above it are inverted into huge numbers and amplify noise. Truncating them, or ridge regularisation (replacing 1/σ by σ/(σ² + λ)), is more stable.", "A⁺ is not continuous in A: a tiny perturbation that changes the rank changes A⁺ drastically."],
    worked: {
      title: "one equation, two unknowns",
      steps: [
        { tex: r`x_1+2x_2=4\quad\Rightarrow\quad A=\begin{bmatrix}1&2\end{bmatrix},\ \ b=4`, note: "infinitely many solutions: a line in the plane" },
        { tex: r`A^+=A^\top(AA^\top)^{-1}=\frac15\begin{bmatrix}1\\2\end{bmatrix}=\begin{bmatrix}0.2\\0.4\end{bmatrix}`, note: "full row rank" },
        { tex: r`\mathbf x^+=A^+b=\begin{bmatrix}0.8\\1.6\end{bmatrix},\qquad A\mathbf x^+=0.8+3.2=4,\qquad\|\mathbf x^+\|=1.789`, note: "an exact solution" },
        { tex: r`\text{another solution }(4,0):\ \|(4,0)\|=4;\qquad(0,2):\ \|(0,2)\|=2`, note: "both longer" },
        { tex: r`\mathbf x^+=0.8\,(1,2)\in C(A^\top),\qquad(4,0)-\mathbf x^+=(3.2,-1.6)\in N(A)`, note: "x⁺ has no null-space component; other solutions differ from it by null-space vectors" },
      ],
      result: r`\mathbf x^+=(0.8,\ 1.6)`,
    },
    applications: [
      { title: "numpy.linalg.lstsq and pinv", body: "Library least-squares routines compute the SVD and return A⁺b, so they succeed even when features are collinear, silently choosing the minimum-norm weights." },
      { title: "Implicit bias of gradient descent", body: "For an over-parameterised linear model, gradient descent started at zero converges to the pseudoinverse solution: of all the weight vectors that fit the data exactly, it finds the smallest." },
    ],
  },
};

export const additionalLinearAlgebra: SectionDef = {
  id: "additional-math",
  part: "additional",
  title: "Linear Algebra Additions",
  blurb: "Four lessons that the foundations and the course weeks rely on but which were not taught on their own: the null space and its three companion subspaces, projection onto a subspace, the spectral theorem and the pseudoinverse. Matrix calculus was added to Calculus, joint and marginal distributions to Probability & Statistics, and convex sets to Optimization.",
  topics: [fourSubspaces, projectionMatrices, spectralTheorem, pseudoinverse],
};

/* ------------------------------------------------------------------ Calculus */
export const matrixCalculus: TopicDef = {
  slug: "matrix-calculus",
  title: "Matrix Calculus: Gradients of Vector Expressions",
  summary: "Differentiate aᵀx, xᵀAx and ‖Ax − b‖² without writing out components.",
  difficulty: "intermediate",
  keywords: ["matrix calculus", "vector calculus", "gradient of quadratic form", "gradient identities", "derivative of x^T A x", "derivative of least squares", "denominator layout"],
  prerequisites: ["calculus/gradient", "calculus/hessian", "linear-algebra/matrix-vector-multiplication", "linear-algebra/transpose"],
  related: ["calculus/jacobian", "linear-algebra/least-squares", "linear-algebra/mlt-linear-regression", "linear-algebra/positive-definite"],
  content: {
    intuition: "Losses in machine learning are written with vectors and matrices, and differentiating them one component at a time is slow and error-prone. A handful of identities — the vector analogues of d(ax)/dx = a and d(ax²)/dx = 2ax — let you differentiate whole expressions at once.",
    definition: "For a scalar function f of a vector x ∈ ℝⁿ, the gradient ∇f is the vector of partial derivatives ∂f/∂xᵢ. The basic identities are: ∇(aᵀx) = a; ∇(xᵀAx) = (A + Aᵀ)x, which is 2Ax when A is symmetric; and ∇‖Ax − b‖² = 2Aᵀ(Ax − b). The Hessians are 0, A + Aᵀ and 2AᵀA respectively.",
    why: "These three identities derive the normal equations of linear regression, the ridge solution, the PCA eigenvalue condition and the gradients used by every optimiser, each in a line or two.",
    usedIn: ["Deriving closed-form solutions (least squares, ridge)", "Gradients for gradient descent on linear and quadratic objectives", "Second-order conditions via Hessians of quadratic forms"],
    visual: { id: "matrix-calculus", caption: "Choose a function, edit A and the vector, and drag x. The gradient from the identity is compared with an independent finite-difference estimate; the arrow shows that it is perpendicular to the level set through x." },
    formulas: [
      { label: "Linear function", tex: r`\nabla_{\mathbf x}\,(\mathbf a^\top\mathbf x)=\mathbf a`, explanation: "The slope of a linear function is its coefficient vector.", variables: [{ symbol: r`\mathbf a`, meaning: "constant vector" }, { symbol: r`\mathbf x\in\mathbb R^n`, meaning: "the variable" }] },
      { label: "Quadratic form", tex: r`\nabla_{\mathbf x}\,(\mathbf x^\top A\mathbf x)=(A+A^\top)\,\mathbf x\ \ \overset{A=A^\top}{=}\ \ 2A\mathbf x`, explanation: "Only the symmetric part of A matters.", variables: [{ symbol: "A", meaning: "constant n × n matrix" }] },
      { label: "Squared norm", tex: r`\nabla_{\mathbf x}\,\|\mathbf x\|^2=2\mathbf x`, explanation: "The case A = I." },
      { label: "Least squares", tex: r`\nabla_{\mathbf x}\,\|A\mathbf x-\mathbf b\|^2=2A^\top(A\mathbf x-\mathbf b),\qquad\nabla^2=2A^\top A`, explanation: "Residual, sent back through Aᵀ. Setting it to zero gives the normal equations." },
      { label: "Chain rule for a linear change of variable", tex: r`\nabla_{\mathbf x}\,g(A\mathbf x)=A^\top\,\nabla g(\mathbf u)\big|_{\mathbf u=A\mathbf x}`, explanation: "Gradients are pulled back through the transpose — the rule backpropagation applies at every linear layer." },
    ],
    derivation: {
      title: "the gradient of xᵀAx",
      steps: [
        { tex: r`f(\mathbf x)=\mathbf x^\top A\mathbf x=\sum_{i}\sum_{j}a_{ij}\,x_ix_j`, note: "Write the quadratic form as a double sum." },
        { tex: r`\frac{\partial f}{\partial x_k}=\sum_ja_{kj}x_j+\sum_ia_{ik}x_i`, note: "xₖ appears once as xᵢ (terms with i = k) and once as xⱼ (terms with j = k); the product rule gives both sums." },
        { tex: r`=(A\mathbf x)_k+(A^\top\mathbf x)_k`, note: "Recognise the k-th entries of Ax and Aᵀx." },
        { tex: r`\nabla f=(A+A^\top)\mathbf x`, note: "Collect over k." },
        { tex: r`\|A\mathbf x-\mathbf b\|^2=\mathbf x^\top A^\top A\mathbf x-2\mathbf b^\top A\mathbf x+\mathbf b^\top\mathbf b\ \Rightarrow\ \nabla=2A^\top A\mathbf x-2A^\top\mathbf b`, note: "Apply the quadratic and linear rules to the expanded least-squares objective (AᵀA is symmetric)." },
      ],
      conclusion: "Every identity can be checked this way, or numerically with finite differences — a habit worth keeping whenever a gradient is derived by hand.",
    },
    assumptions: ["Gradients are written as column vectors of the same shape as x (the 'denominator layout'). Some texts use the transpose convention; the results differ only by a transpose."],
    specialCases: ["If A is antisymmetric (Aᵀ = −A) then xᵀAx = 0 for all x and the gradient vanishes.", "For symmetric positive definite A, ∇(½xᵀAx − bᵀx) = Ax − b, so minimising the quadratic is solving Ax = b."],
    limitations: ["These identities cover scalar functions of a vector. Derivatives with respect to matrices, or of vector-valued functions, need Jacobians and more careful index bookkeeping."],
    worked: {
      title: "checking the quadratic-form rule numerically",
      steps: [
        { tex: r`A=\begin{bmatrix}2&1\\0&1\end{bmatrix},\quad\mathbf x=(1,2):\qquad f(\mathbf x)=2x_1^2+x_1x_2+x_2^2=2+2+4=8`, note: "a non-symmetric A" },
        { tex: r`(A+A^\top)\mathbf x=\begin{bmatrix}4&1\\1&2\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix}=\begin{bmatrix}6\\5\end{bmatrix}`, note: "the identity" },
        { tex: r`\frac{\partial f}{\partial x_1}=4x_1+x_2=6,\qquad\frac{\partial f}{\partial x_2}=x_1+2x_2=5`, note: "component-wise differentiation agrees" },
        { tex: r`2A\mathbf x=(8,4)\neq(6,5)`, note: "the shortcut 2Ax is wrong here because A is not symmetric" },
      ],
      result: r`\nabla f(1,2)=(6,\ 5)`,
    },
    applications: [
      { title: "Normal equations in one line", body: "With L(w) = ½‖Xᵀw − y‖², the least-squares identity gives ∇L = X(Xᵀw − y); setting it to zero is the normal equation XXᵀw = Xy." },
      { title: "Gradient checking", body: "Comparing an analytic gradient with central finite differences, as the visualisation does, is the standard test for hand-written backward passes." },
    ],
  },
};

/* ------------------------------------------------------------------ Probability and statistics */
export const jointMarginal: TopicDef = {
  slug: "joint-marginal-distributions",
  title: "Joint, Marginal and Conditional Distributions",
  summary: "One table for two random variables, and the distributions that can be read off it.",
  difficulty: "beginner",
  keywords: ["joint distribution", "marginal distribution", "conditional distribution", "marginalization", "sum rule", "product rule", "independence", "contingency table"],
  prerequisites: ["statistics/random-variables", "statistics/conditional-probability"],
  related: ["statistics/bayes-theorem", "statistics/covariance-correlation", "statistics/conditional-independence", "linear-algebra/mlt-generative-discriminative"],
  content: {
    intuition: "Record two things about every customer: age group and whether they bought. A table of how often each combination occurs is the joint distribution. Adding up a row or a column forgets one variable and leaves the marginal distribution of the other. Looking at a single row, rescaled to sum to one, tells you about one variable given a value of the other.",
    definition: "The joint distribution of X and Y gives P(X = x, Y = y) for every pair of values. The marginal distribution of X is obtained by summing the joint over y (the sum rule). The conditional distribution of X given Y = y is the joint divided by the marginal P(Y = y) (the product rule rearranged). X and Y are independent when the joint equals the product of the marginals for every pair.",
    why: "All of probabilistic machine learning is manipulation of joint distributions with these two rules. Generative classifiers model the joint of features and label; Bayesian inference marginalises over parameters; independence assumptions are what make large joints tractable.",
    usedIn: ["Generative models such as Naive Bayes and GMMs", "Bayes' theorem, which combines the two rules", "Mutual information and feature selection", "Marginalising latent variables in EM and variational methods"],
    visual: { id: "joint-marginal", caption: "A joint table for X ∈ {1, …, 4} and Y ∈ {1, 2, 3}. The margins are its row and column sums; click a cell to condition on its row. The slider moves the table from independent to strongly dependent, and the readouts measure the departure from independence." },
    formulas: [
      { label: "Joint distribution", tex: r`P(X=x,\,Y=y)\ge0,\qquad\sum_x\sum_yP(X=x,\,Y=y)=1`, explanation: "A probability for every combination of values.", variables: [{ symbol: "X,\\ Y", meaning: "two random variables" }] },
      { label: "Marginalisation (sum rule)", tex: r`P(X=x)=\sum_yP(X=x,\,Y=y)`, explanation: "Sum out the variable you do not care about. For continuous variables the sum becomes an integral." },
      { label: "Conditional distribution", tex: r`P(X=x\mid Y=y)=\frac{P(X=x,\,Y=y)}{P(Y=y)}`, explanation: "A row (or column) of the joint, renormalised to sum to one." },
      { label: "Product rule", tex: r`P(X=x,\,Y=y)=P(X=x\mid Y=y)\,P(Y=y)`, explanation: "Any joint factorises into a marginal and a conditional." },
      { label: "Independence", tex: r`X\perp Y\iff P(X=x,\,Y=y)=P(X=x)\,P(Y=y)\ \ \forall x,y`, explanation: "Equivalently, every conditional equals the marginal: knowing Y tells you nothing about X." },
    ],
    derivation: {
      title: "Bayes' theorem from the two rules",
      steps: [
        { tex: r`P(x,y)=P(x\mid y)\,P(y)=P(y\mid x)\,P(x)`, note: "The product rule, applied in both orders." },
        { tex: r`P(y\mid x)=\frac{P(x\mid y)\,P(y)}{P(x)}`, note: "Divide by P(x)." },
        { tex: r`P(x)=\sum_{y'}P(x,y')=\sum_{y'}P(x\mid y')\,P(y')`, note: "The denominator is a marginal, obtained with the sum rule." },
        { tex: r`P(y\mid x)=\frac{P(x\mid y)\,P(y)}{\sum_{y'}P(x\mid y')\,P(y')}`, note: "Bayes' theorem." },
      ],
      conclusion: "The sum rule and the product rule are the only rules of probability needed; everything else, including Bayes' theorem, is a consequence.",
    },
    assumptions: ["Conditioning on Y = y requires P(Y = y) > 0."],
    specialCases: ["If X and Y are independent, the joint table is the outer product of the two marginal vectors and has rank one.", "Zero covariance does not imply independence: the joint can have structure that the covariance cannot see."],
    limitations: ["A joint table over d binary variables has 2ᵈ entries. Storing or estimating it directly is impossible beyond a few variables, which is why models impose structure such as conditional independence.", "The marginals do not determine the joint: many different joints share the same row and column sums."],
    worked: {
      title: "a 2 × 2 table",
      steps: [
        { tex: r`\begin{array}{c|cc}&Y=0&Y=1\\\hline X=0&0.1&0.3\\X=1&0.2&0.4\end{array}`, note: "joint probabilities; they sum to 1" },
        { tex: r`P(X=0)=0.1+0.3=0.4,\quad P(X=1)=0.6;\qquad P(Y=0)=0.3,\quad P(Y=1)=0.7`, note: "row sums and column sums" },
        { tex: r`P(X=1\mid Y=1)=\frac{0.4}{0.7}=0.571`, note: "condition on the second column" },
        { tex: r`P(X=1)\,P(Y=1)=0.6\times0.7=0.42\neq0.4`, note: "the product of the marginals differs from the joint entry: X and Y are dependent" },
        { tex: r`P(X=1\mid Y=1)=0.571<P(X=1)=0.6`, note: "the same conclusion: learning Y = 1 lowers the probability of X = 1" },
      ],
      result: r`P(X=1\mid Y=1)=\tfrac47`,
    },
    applications: [
      { title: "Generative classification", body: "Naive Bayes stores the joint P(x, y) in factored form, P(y)ΠⱼP(xⱼ | y), and classifies by computing the conditional P(y | x) — a row normalisation of the joint." },
      { title: "Latent variables", body: "A Gaussian mixture defines a joint over a hidden component z and the observation x; the density of x alone is its marginal Σ_z P(z)P(x | z), and the E-step computes the conditional P(z | x)." },
    ],
  },
};

/* ------------------------------------------------------------------ Optimization */
export const convexSets: TopicDef = {
  slug: "convex-sets",
  title: "Convex Sets",
  summary: "Sets that contain the whole segment between any two of their points.",
  difficulty: "intermediate",
  keywords: ["convex set", "line segment", "convex combination", "half-space", "norm ball", "polyhedron", "intersection of convex sets", "feasible region"],
  prerequisites: ["linear-algebra/linear-combinations", "linear-algebra/norms"],
  related: ["calculus/convex-functions", "optimization/convex-optimization", "optimization/constrained-optimization", "linear-algebra/mlt-svm-margins"],
  content: {
    intuition: "A set is convex if, standing at any point of it, you can see every other point: no dents, no holes, no separate pieces. Walk in a straight line between any two points and you never step outside. A disc is convex; a crescent is not.",
    definition: "A set S ⊆ ℝⁿ is convex if for all a, b ∈ S and all θ ∈ [0, 1], the point θa + (1 − θ)b is also in S. Half-spaces {x : wᵀx ≤ c}, norm balls {x : ‖x‖ ≤ ρ} and affine subspaces are convex, and any intersection of convex sets is convex.",
    why: "Convex optimisation means minimising a convex function over a convex set, and it is the convexity of the feasible set — together with that of the objective — that rules out spurious local minima. The constraint sets of ridge, LASSO and the SVM are all convex.",
    usedIn: ["Feasible regions of constrained optimisation problems", "Norm-ball constraints in regularisation", "SVM constraints (an intersection of half-spaces)", "The probability simplex of valid distributions"],
    visual: { id: "convex-sets", caption: "Pick a set and drag the two points. The part of the connecting segment that leaves the set is drawn in red. A Monte-Carlo readout tries 2000 random pairs of points of the set and reports how many segments escape." },
    formulas: [
      { label: "Definition", tex: r`\mathbf a,\mathbf b\in S,\ \ \theta\in[0,1]\ \Longrightarrow\ \theta\mathbf a+(1-\theta)\mathbf b\in S`, explanation: "The segment between any two points of S stays in S.", variables: [{ symbol: r`\theta`, meaning: "position along the segment: 1 at a, 0 at b" }] },
      { label: "Convex combination", tex: r`\sum_{i=1}^{k}\theta_i\mathbf x_i,\qquad\theta_i\ge0,\ \ \sum_i\theta_i=1`, explanation: "A weighted average of points. A convex set contains every convex combination of its points." },
      { label: "Half-space and hyperplane", tex: r`\{\mathbf x:\mathbf w^\top\mathbf x\le c\},\qquad\{\mathbf x:\mathbf w^\top\mathbf x=c\}`, explanation: "The basic convex sets; a linear inequality or equality constraint." },
      { label: "Norm ball", tex: r`\{\mathbf x:\|\mathbf x\|\le\rho\}`, explanation: "Convex for every norm, by the triangle inequality: the L2 disc, the L1 diamond, the L∞ square." },
      { label: "Intersection", tex: r`S_1,S_2\ \text{convex}\ \Longrightarrow\ S_1\cap S_2\ \text{convex}`, explanation: "Several convex constraints together still define a convex feasible set. Unions, in general, do not." },
    ],
    derivation: {
      title: "half-spaces and their intersections are convex",
      steps: [
        { tex: r`\mathbf w^\top\mathbf a\le c,\qquad\mathbf w^\top\mathbf b\le c`, note: "Take two points of the half-space." },
        { tex: r`\mathbf w^\top\big(\theta\mathbf a+(1-\theta)\mathbf b\big)=\theta\,\mathbf w^\top\mathbf a+(1-\theta)\,\mathbf w^\top\mathbf b\le\theta c+(1-\theta)c=c`, note: "Linearity, and θ, 1 − θ ≥ 0 preserve the inequalities." },
        { tex: r`\mathbf a,\mathbf b\in S_1\cap S_2\ \Rightarrow\ \theta\mathbf a+(1-\theta)\mathbf b\in S_1\ \text{and}\ \in S_2`, note: "Both points lie in each set, and each set is convex." },
        { tex: r`\Rightarrow\ \theta\mathbf a+(1-\theta)\mathbf b\in S_1\cap S_2`, note: "So the intersection is convex. The argument works for any number of sets." },
        { tex: r`\|\theta\mathbf a+(1-\theta)\mathbf b\|\le\theta\|\mathbf a\|+(1-\theta)\|\mathbf b\|\le\rho`, note: "Norm balls: triangle inequality and homogeneity." },
      ],
      conclusion: "Every polyhedron (a region cut out by finitely many linear inequalities) is convex, as is its intersection with any norm ball.",
    },
    specialCases: ["The empty set, a single point and the whole space are convex.", "The sub-level set {x : f(x) ≤ c} of a convex function f is convex — the link between convex functions and convex sets."],
    limitations: ["Convexity of the feasible set alone guarantees nothing; the objective must be convex too.", "Many constraint sets of practical interest are not convex — for example matrices of rank at most r, or vectors with at most k non-zero entries — which is why their convex surrogates (nuclear norm, L1 norm) are used."],
    worked: {
      title: "testing two sets",
      steps: [
        { tex: r`S=\{\mathbf x:\|\mathbf x\|_2\le2\},\quad\mathbf a=(2,0),\ \mathbf b=(0,2),\ \theta=\tfrac12\ \Rightarrow\ (1,1),\ \ \|(1,1)\|=1.414\le2`, note: "the midpoint is inside, as it must be for a disc" },
        { tex: r`T=\{\mathbf x:1\le\|\mathbf x\|_2\le2\}\ (\text{a ring}),\quad\mathbf a=(1.5,0),\ \mathbf b=(-1.5,0)`, note: "both points belong to the ring" },
        { tex: r`\theta=\tfrac12:\ (0,0),\qquad\|(0,0)\|=0<1\ \Rightarrow\ (0,0)\notin T`, note: "the midpoint falls in the hole: one counter-example proves T is not convex" },
        { tex: r`\{\mathbf x:x_1+x_2\le2\}\cap\{\mathbf x:x_1\ge0\}\cap\{\mathbf x:x_2\ge0\}`, note: "a triangle: an intersection of three half-spaces, hence convex" },
      ],
      result: r`S\ \text{convex},\qquad T\ \text{not convex}`,
    },
    applications: [
      { title: "Regularisation as a constraint", body: "Ridge and LASSO can be written as minimising the loss over an L2 or L1 ball. Both balls are convex, so both problems are convex; the L1 ball's corners are what produce sparse solutions." },
      { title: "SVM feasibility", body: "Each margin constraint yᵢwᵀxᵢ ≥ 1 is a half-space in w. Their intersection is a convex polyhedron, over which the convex objective ½‖w‖² has a unique minimiser." },
    ],
  },
};
