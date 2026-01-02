import numpy as np
from typing import Union, Tuple, Optional, List
import scipy.linalg as la
from scipy.optimize import minimize
import warnings


class MatrixOperations:
    """Core matrix operations for custom ML implementations without external ML libraries."""
    
    @staticmethod
    def stable_inverse(matrix: np.ndarray, rcond: float = 1e-15) -> np.ndarray:
        """
        Compute stable matrix inverse using SVD decomposition.
        
        Args:
            matrix: Square matrix to invert
            rcond: Cutoff for small singular values
            
        Returns:
            Inverse of the matrix
        """
        try:
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
            s_inv = np.zeros_like(s)
            s_inv[s > rcond] = 1.0 / s[s > rcond]
            return Vh.T @ np.diag(s_inv) @ U.T
        except np.linalg.LinAlgError:
            warnings.warn("SVD did not converge, using pseudo-inverse")
            return np.linalg.pinv(matrix, rcond=rcond)
    
    @staticmethod
    def cholesky_safe(matrix: np.ndarray, add_diagonal: float = 1e-6) -> np.ndarray:
        """
        Safe Cholesky decomposition with regularization for nearly singular matrices.
        
        Args:
            matrix: Positive semi-definite matrix
            add_diagonal: Small value to add to diagonal for numerical stability
            
        Returns:
            Lower triangular Cholesky factor
        """
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            # Add small diagonal perturbation
            n = matrix.shape[0]
            perturbed = matrix + add_diagonal * np.eye(n)
            try:
                return np.linalg.cholesky(perturbed)
            except np.linalg.LinAlgError:
                # If still fails, use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(matrix)
                eigvals = np.maximum(eigvals, add_diagonal)
                return eigvecs @ np.diag(np.sqrt(eigvals))
    
    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray, method: str = 'lu') -> np.ndarray:
        """
        Solve linear system Ax = b using specified method.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            method: 'lu', 'qr', or 'svd'
            
        Returns:
            Solution vector x
        """
        if method == 'lu':
            try:
                lu, piv = la.lu_factor(A)
                return la.lu_solve((lu, piv), b)
            except:
                method = 'qr'  # Fallback to QR
                
        if method == 'qr':
            Q, R = np.linalg.qr(A)
            return np.linalg.solve(R, Q.T @ b)
            
        elif method == 'svd':
            return MatrixOperations.stable_inverse(A) @ b
    
    @staticmethod
    def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
        """
        Gram-Schmidt orthogonalization process.
        
        Args:
            vectors: Matrix with columns as vectors to orthogonalize
            
        Returns:
            Matrix with orthonormal columns
        """
        n, m = vectors.shape
        Q = np.zeros((n, m))
        
        for j in range(m):
            q = vectors[:, j].copy()
            for i in range(j):
                q -= np.dot(Q[:, i], vectors[:, j]) * Q[:, i]
            
            norm = np.linalg.norm(q)
            if norm > 1e-10:
                Q[:, j] = q / norm
            else:
                # Handle linear dependence
                Q[:, j] = 0
                
        return Q
    
    @staticmethod
    def matrix_exponential(A: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Compute matrix exponential e^(tA) using Padé approximation.
        
        Args:
            A: Square matrix
            t: Time parameter
            
        Returns:
            Matrix exponential
        """
        return la.expm(t * A)
    
    @staticmethod
    def power_iteration(A: np.ndarray, num_iterations: int = 100, 
                       tolerance: float = 1e-10) -> Tuple[float, np.ndarray]:
        """
        Power iteration to find largest eigenvalue and eigenvector.
        
        Args:
            A: Square matrix
            num_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (eigenvalue, eigenvector)
        """
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        eigenvalue = 0
        for _ in range(num_iterations):
            Av = A @ v
            eigenvalue_new = np.dot(v, Av)
            v_new = Av / np.linalg.norm(Av)
            
            if np.abs(eigenvalue_new - eigenvalue) < tolerance:
                break
                
            eigenvalue = eigenvalue_new
            v = v_new
            
        return eigenvalue, v
    
    @staticmethod
    def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute Kronecker product of two matrices.
        
        Args:
            A: First matrix
            B: Second matrix
            
        Returns:
            Kronecker product A ⊗ B
        """
        return np.kron(A, B)
    
    @staticmethod
    def vec(matrix: np.ndarray) -> np.ndarray:
        """
        Vectorize a matrix by stacking columns.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Vectorized form
        """
        return matrix.flatten('F')  # Fortran order (column-major)
    
    @staticmethod
    def unvec(vector: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Reshape vector back to matrix form.
        
        Args:
            vector: Vectorized matrix
            shape: Original matrix shape (rows, cols)
            
        Returns:
            Matrix form
        """
        return vector.reshape(shape, order='F')
    
    @staticmethod
    def sylvester_solve(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Solve Sylvester equation AX + XB = C.
        
        Args:
            A, B, C: Matrix coefficients
            
        Returns:
            Solution matrix X
        """
        return la.solve_sylvester(A, B, C)
    
    @staticmethod
    def matrix_sqrt(A: np.ndarray, method: str = 'diag') -> np.ndarray:
        """
        Compute matrix square root.
        
        Args:
            A: Positive semi-definite matrix
            method: 'diag' for eigendecomposition, 'schur' for Schur method
            
        Returns:
            Matrix square root
        """
        if method == 'diag':
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.maximum(eigvals, 0)  # Handle numerical errors
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        else:
            return la.sqrtm(A)
    
    @staticmethod
    def robust_covariance(data: np.ndarray, method: str = 'empirical',
                         shrinkage: float = 0.1) -> np.ndarray:
        """
        Compute robust covariance matrix.
        
        Args:
            data: Data matrix (n_samples, n_features)
            method: 'empirical', 'oas' (Oracle Approximating Shrinkage), or 'ledoit-wolf'
            shrinkage: Shrinkage parameter for regularization
            
        Returns:
            Covariance matrix
        """
        n_samples, n_features = data.shape
        
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        
        if method == 'empirical':
            cov = (data_centered.T @ data_centered) / (n_samples - 1)
        
        elif method == 'oas' or method == 'ledoit-wolf':
            # Empirical covariance
            emp_cov = (data_centered.T @ data_centered) / (n_samples - 1)
            
            # Shrinkage target (diagonal matrix with average variance)
            mu = np.trace(emp_cov) / n_features
            target = mu * np.eye(n_features)
            
            if method == 'oas':
                # Oracle Approximating Shrinkage
                alpha = (1 - 2/n_features) / n_samples
                beta = 1 / (n_samples * (n_samples + 1 - 2/n_features))
                
                # Compute shrinkage coefficient
                delta = np.linalg.norm(emp_cov - target, 'fro') ** 2
                shrinkage = min(alpha + beta * delta / mu**2, 1)
            
            cov = (1 - shrinkage) * emp_cov + shrinkage * target
            
        return cov
    
    @staticmethod
    def numerical_gradient(func, x: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
        """
        Compute numerical gradient using central differences.
        
        Args:
            func: Function to differentiate
            x: Point at which to evaluate gradient
            epsilon: Small perturbation for finite differences
            
        Returns:
            Gradient vector
        """
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
            
        return grad
    
    @staticmethod
    def numerical_hessian(func, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute numerical Hessian matrix using finite differences.
        
        Args:
            func: Function to differentiate twice
            x: Point at which to evaluate Hessian
            epsilon: Small perturbation for finite differences
            
        Returns:
            Hessian matrix
        """
        n = len(x)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += epsilon
                x_pp[j] += epsilon
                x_pm[i] += epsilon
                x_pm[j] -= epsilon
                x_mp[i] -= epsilon
                x_mp[j] += epsilon
                x_mm[i] -= epsilon
                x_mm[j] -= epsilon
                
                hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * epsilon**2)
                hess[j, i] = hess[i, j]
                
        return hess
    
    @staticmethod
    def line_search(func, x: np.ndarray, direction: np.ndarray,
                   alpha_init: float = 1.0, c1: float = 1e-4,
                   c2: float = 0.9, max_iter: int = 20) -> float:
        """
        Backtracking line search with Wolfe conditions.
        
        Args:
            func: Objective function
            x: Current point
            direction: Search direction
            alpha_init: Initial step size
            c1: Armijo condition parameter
            c2: Curvature condition parameter
            max_iter: Maximum iterations
            
        Returns:
            Optimal step size
        """
        alpha = alpha_init
        f0 = func(x)
        grad0 = MatrixOperations.numerical_gradient(func, x)
        
        for _ in range(max_iter):
            x_new = x + alpha * direction
            f_new = func(x_new)
            
            # Armijo condition
            if f_new <= f0 + c1 * alpha * np.dot(grad0, direction):
                # Curvature condition (strong Wolfe)
                grad_new = MatrixOperations.numerical_gradient(func, x_new)
                if np.abs(np.dot(grad_new, direction)) <= c2 * np.abs(np.dot(grad0, direction)):
                    return alpha
                    
            alpha *= 0.5
            
        return alpha
    
    @staticmethod
    def conjugate_gradient(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                          tol: float = 1e-10, max_iter: Optional[int] = None) -> np.ndarray:
        """
        Conjugate gradient method for solving Ax = b.
        
        Args:
            A: Symmetric positive-definite matrix
            b: Right-hand side vector
            x0: Initial guess
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Solution vector
        """
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
            
        if max_iter is None:
            max_iter = n
            
        r = b - A @ x
        p = r.copy()
        rsold = np.dot(r, r)
        
        for _ in range(max_iter):
            if rsold < tol**2:
                break
                
            Ap = A @ p
            alpha = rsold / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = np.dot(r, r)
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
            
        return x


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding operator for L1 regularization.
    
    Args:
        x: Input array
        threshold: Threshold value
        
    Returns:
        Soft-thresholded array
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def proximal_gradient(func, grad_func, prox_func, x0: np.ndarray,
                     step_size: float = 0.01, max_iter: int = 1000,
                     tol: float = 1e-6) -> np.ndarray:
    """
    Proximal gradient method for composite optimization.
    
    Args:
        func: Smooth objective function
        grad_func: Gradient of smooth part
        prox_func: Proximal operator for non-smooth part
        x0: Initial point
        step_size: Step size (learning rate)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Optimal point
    """
    x = x0.copy()
    
    for i in range(max_iter):
        grad = grad_func(x)
        x_new = prox_func(x - step_size * grad, step_size)
        
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        
    return x


def newton_raphson(func, grad_func, hess_func, x0: np.ndarray,
                  tol: float = 1e-6, max_iter: int = 100,
                  line_search: bool = True) -> np.ndarray:
    """
    Newton-Raphson method for optimization.
    
    Args:
        func: Objective function
        grad_func: Gradient function
        hess_func: Hessian function
        x0: Initial point
        tol: Convergence tolerance
        max_iter: Maximum iterations
        line_search: Whether to use line search
        
    Returns:
        Optimal point
    """
    x = x0.copy()
    
    for i in range(max_iter):
        grad = grad_func(x)
        hess = hess_func(x)
        
        # Newton direction
        try:
            direction = -MatrixOperations.solve_linear_system(hess, grad)
        except:
            # If Hessian is singular, use gradient descent
            direction = -grad
            
        if line_search:
            alpha = MatrixOperations.line_search(func, x, direction)
        else:
            alpha = 1.0
            
        x_new = x + alpha * direction
        
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        
    return x