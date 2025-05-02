from pathlib import Path
import numpy as np
from stl import mesh
import math

class GeometricOperator:
    """
    A class to compute geometric operators and operator vectors from STL geometry.
    """

    def __init__(self, stl_geometry):
        """
        Initialize the GeometricOperator with STL geometry.

        Parameters:
        - stl_geometry (mesh.Mesh): The STL geometry data.

        Attributes:
        - stlGeometry (mesh.Mesh): The STL geometry data.
        - volume (float): Volume of the STL geometry.
        - centroid (array-like): Centroid of the STL geometry.
        - hullTriangles (array-like): Triangles of the STL geometry as float.
        """
        self.stlGeometry = stl_geometry
        self.volume, self.centroid, _ = stl_geometry.get_mass_properties()
        self.hullTriangles = self.stlGeometry.vectors.astype(float)

    def getOperator(self, p, q, r, is_centered, is_scaled):
        """
        Calculate the operator based on given parameters.

        Parameters:
        - p, q, r (int): operator order parameters.
        - is_centered (int): If 1, triangles are centered using the centroid. Otherwise, they remain unchanged.
        - is_scaled (int): If 1, the moment is scaled by the volume.

        Returns:
        - operator (float): The computed moment value.
        """
        s = p + q + r
        # Create combinations of p, q, r
        k1 = np.array([(a, b, p - a - b) for a in range(10) for b in range(10) if 0 <= p - a - b < 10])
        k2 = np.array([(a, b, q - a - b) for a in range(10) for b in range(10) if 0 <= q - a - b < 10])
        k3 = np.array([(a, b, r - a - b) for a in range(10) for b in range(10) if 0 <= r - a - b < 10])

        # Combine the combinations to form an array of shape (-1, 3, 3)
        s_out = np.array([[[i, j, k] for i in k1 for j in k2 for k in k3]]).reshape(-1, 3, 3)

        # Check if triangles should be centered or not
        if is_centered == 1:
            triangles = self.hullTriangles - self.centroid
        else:
            triangles = self.hullTriangles

        # Calculate the first term using determinants and factorials
        term1 = np.linalg.det(triangles) *   math.factorial(p) * math.factorial(q) * math.factorial(
            r) / math.factorial(s + 3)

        # Auxiliary calculations for the second term
        k_sums = s_out.sum(axis=1)
        factorials = np.vectorize(math.factorial)
        numerator = factorials(k_sums[:, 0]) * factorials(k_sums[:, 1]) * factorials(k_sums[:, 2])
        denominator = np.prod(factorials(s_out.reshape(-1, 3)), axis=1).reshape(-1, 3).prod(axis=1)
        triangles = triangles.transpose(0, 2, 1)
        exterm_right_term = np.prod(triangles[:, None, :, :] ** s_out[None, :, :, :], axis=(2, 3))

        # Compute the second term
        term2 = ((numerator / denominator) * exterm_right_term).sum(axis=1)

        # Final calculation of the operator
        operator = (term1 * term2).sum()
        if is_scaled == 1:
            operator = operator / (self.volume ** (1 + (s / 3.0)))

        return operator

    # Calculate a vector of moments based on a given order
    def getOperatorVector(self, order, is_centered, is_scaled):
        """
        Calculate a vector of operators based on a given order.

        Parameters:
        - order (int): The order for the operator vector.
        - is_centered (int): If 1, triangles are centered using the centroid. Otherwise, they remain unchanged.
        - is_scaled (int): If 1, the operator is scaled by the volume.

        Returns:
        - operator_vector (array-like): The computed operator vector.
        - operator_names (array-like): Names corresponding to each operator in the vector.
        """
        # Create combinations of p, q, r based on the given order
        pqr = np.array([(a, b, order - a - b) for a in range(10) for b in range(10) if 0 <= order - a - b < 10])

        # Calculate the operator vector using the getMoment function
        operator_vector = np.array([self.getOperator(p, q, r, is_centered, is_scaled) for p, q, r in pqr])

        # Generate names for the operators
        operator_names = np.array([f"M^{p},{q},{r}" for p, q, r in pqr])

        return operator_vector, operator_names


def main():
    all_operators = []

    for design_num in range(1):
        print(f"Processing Design: {design_num}")

        design_path = Path(r"M:\My Documents\KirstenOdendaal\70180 - Ship Hull GAN\Geometries\DSYH\dsy_geo") / f"Ship_{design_num}.stl"

        try:
            stl_data = mesh.Mesh.from_file(design_path)
            gm = GeometricOperator(stl_data)

            operator_vector_list = []
            for order in range(5):
                operator_vector_, _ = gm.getOperatorVector(order, 0, 0)  # order, isCentered, isScaled
                operator_vector_list.extend(operator_vector_)
                print(_)
            operator_vector = np.array(operator_vector_list)
            all_operators.append(operator_vector)

            save_path = Path(r"M:\My Documents\KirstenOdendaal\70180 - Ship Hull GAN\Geometries\DSYH\operators") / f"operators_{design_num}.csv"
            np.savetxt(save_path, operator_vector, delimiter=',')
        except FileNotFoundError:
            print(f"Design file {design_path} not found.")
            continue

    combined_save_path = r"M:\My Documents\KirstenOdendaal\70180 - Ship Hull GAN\Geometries\DSYH\operators\operators_all.csv"
    np.savetxt(combined_save_path, np.array(all_operators), delimiter=',')

if __name__ == '__main__':
    main()
