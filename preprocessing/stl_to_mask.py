"""
Turn an STL bone surface into a voxel mask on the same grid as its CT scan.

"""
import argparse

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


def read_stl(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def stl_to_mask(ct, mesh, ras=False):
    """Fill the closed surface `mesh` into a 0/1 mask with the same grid as `ct`."""
    # 1. Mesh points, in millimetres (physical space)
    pts = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    if ras:
        # RAS -> LPS (ITK's convention): flip the sign of x and y
        pts[:, 0] *= -1
        pts[:, 1] *= -1

    # 2. Millimetres -> voxel index coordinates of the CT
    #    index = inverse(Direction * diag(Spacing)) @ (point - Origin)
    origin = np.array(ct.GetOrigin())
    spacing = np.array(ct.GetSpacing())
    direction = np.array(ct.GetDirection()).reshape(3, 3)
    to_index = np.linalg.inv(direction @ np.diag(spacing))
    idx = (pts - origin) @ to_index.T

    # 3. Put the moved points back into the mesh
    index_mesh = vtk.vtkPolyData()
    index_mesh.DeepCopy(mesh)
    index_mesh.GetPoints().SetData(numpy_support.numpy_to_vtk(idx, deep=True))

    # 4. Fill the inside of the surface on a grid of unit voxels (= the CT's grid)
    size = ct.GetSize()  # (x, y, z)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(index_mesh)
    stencil.SetOutputOrigin(0, 0, 0)
    stencil.SetOutputSpacing(1, 1, 1)
    stencil.SetOutputWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    stencil.Update()

    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(stencil.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()

    arr = numpy_support.vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars())
    arr = arr.reshape(size[2], size[1], size[0])  # VTK x-fastest -> numpy (z, y, x)

    # 5. Wrap as an ITK image with exactly the CT's geometry
    mask = sitk.GetImageFromArray(arr.astype(np.uint8))
    mask.CopyInformation(ct)
    return mask


def quality_checks(ct, mesh, mask):
    ct_arr = sitk.GetArrayFromImage(ct)
    m = sitk.GetArrayFromImage(mask).astype(bool)
    spacing = np.array(ct.GetSpacing())

    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    open_edges = edges.GetOutput().GetNumberOfCells()

    n = int(m.sum())
    print(f"  surface closed (watertight): {'yes' if open_edges == 0 else f'NO ({open_edges} open edges)'}")
    print(f"  voxels inside mask:          {n}")
    print(f"  bone volume:                 {n * spacing.prod() / 1000:.1f} cm^3")
    if n:
        zs = np.nonzero(m.any(axis=(1, 2)))[0]
        length = (zs.max() - zs.min() + 1) * spacing[2]
        print(f"  mask height (along z):       {length:.0f} mm")
        print(f"  median CT value inside mask: {np.median(ct_arr[m]):.0f}"
              "   (should be bone-like, roughly 200+)")
    if n == 0:
        print("  WARNING: empty mask. The STL is probably outside the CT -> try --ras.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ct")
    p.add_argument("stl")
    p.add_argument("out")
    p.add_argument("--ras", action="store_true", help="STL coordinates are RAS, not LPS")
    a = p.parse_args()

    ct = sitk.ReadImage(a.ct)
    mesh = read_stl(a.stl)
    mask = stl_to_mask(ct, mesh, ras=a.ras)
    sitk.WriteImage(mask, a.out, useCompression=True)
    print(f"Saved {a.out}")
    quality_checks(ct, mesh, mask)


if __name__ == "__main__":
    main()
