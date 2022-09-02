import nrrd
import numpy as np
import tifffile as tiff


def load_data(file_path: str) -> np.ndarray:
    """
    Loads either tiff or nrrd files and returns them as numpy arrays with order (z,y,x).
    """

    if file_path.endswith('.tiff') or file_path.endswith('.tif'):
        return tiff.imread(file_path)
    elif file_path.endswith('.nrrd'):
        return nrrd.read(file_path, index_order='C')[0]
    else:
        raise Exception('Only .tiff or .nrrd images can be read.')


def get_hemisphere(z: int, y: int, x: int, delta: int, z_step: int = 1, out_file: str = None) -> np.ndarray:
    hemisphere = np.zeros((z, y, x), np.uint8)

    d = min(min(x, y) - delta, z*z_step*2)
    r = int(d / 2)

    m_z = z
    m_y = int(y/2)
    m_x = int(x/2)

    # fill inner cube
    a = int(np.sqrt(d**2/3) / 2)
    a_z = int(a/z_step)
    hemisphere[m_z-a_z:m_z, m_y-a:m_y+a, m_x-a:m_x+a] = 1

    for z_i in range(m_z - int(r/z_step), m_z):
        for y_i in range(m_y - r, m_y + r + 1):
            for x_i in range(m_x - r, m_x + r + 1):
                if hemisphere[z_i, y_i, x_i] == 1:
                    continue
                if (x_i - m_x)**2 + (y_i - m_y)**2 + ((z_i - m_z) * z_step)**2 <= r**2:
                    hemisphere[z_i, y_i, x_i] = 1

    if out_file is not None:
        tiff.imwrite(out_file, hemisphere * 255)

    return hemisphere


def get_sphere(z: int, y: int, x: int, delta: int = 0, z_step: int = 1, out_file: str = None) -> np.ndarray:
    sphere = np.zeros((z, y, x), np.uint8)

    d = min(x, y, z * z_step) - delta
    r = int(d / 2)

    m_z = int(z/2)
    m_y = int(y/2)
    m_x = int(x/2)

    # fill inner cube
    a = int(np.sqrt(d**2/3) / 2)
    a_z = int(a/z_step)
    sphere[m_z-a_z:m_z+a_z, m_y-a:m_y+a, m_x-a:m_x+a] = 1

    for z_i in range(m_z - int(r/z_step), m_z + int(r/z_step)):
        for y_i in range(m_y - r, m_y + r):
            for x_i in range(m_x - r, m_x + r):
                if sphere[z_i, y_i, x_i] == 1:
                    continue
                if (x_i - m_x)**2 + (y_i - m_y)**2 + ((z_i - m_z) * z_step)**2 <= r**2:
                    sphere[z_i, y_i, x_i] = 1

    if out_file is not None:
        tiff.imwrite(out_file, sphere * 255)

    return sphere