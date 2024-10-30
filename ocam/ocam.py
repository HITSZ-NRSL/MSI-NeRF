import torch

class OcamCamera:
    def __init__(self, filename, fov=360, show_flag=False):
        with open(filename, "r") as file:
            lines = file.readlines()
        calibdata = []
        for line in lines:
            if (line[0] == '#' or line[0] == '\n'):
                continue
            calibdata.append([float(i) for i in line.split()])

        # polynomial coefficients for the DIRECT mapping function
        self._pol = calibdata[0][1:]
        # polynomial coefficients for the inverse mapping function
        self._invpol = calibdata[1][1:]
        # center: "row" and "column", starting from 0 (C convention)
        self._xc = calibdata[2][0]
        self._yc = calibdata[2][1]
        # _affine parameters "c", "d", "e"
        self._affine = calibdata[3]
        # image size: "height" and "width"
        self._img_size = (int(calibdata[4][0]), int(calibdata[4][1]))

        # field of view
        self._fov = fov

        if show_flag:
            print(self)

    def cam2world(self, point2D):
        assert point2D.shape[1] == 2
        
        invdet = 1 / (self._affine[0] - self._affine[1] * self._affine[2])
        xp = invdet * ((point2D[:, 1] - self._xc) - self._affine[1] * (point2D[:, 0] - self._yc))
        yp = invdet * (-self._affine[2] * (point2D[:, 1] - self._xc) + self._affine[0] * (point2D[:, 0] - self._yc))

        r = torch.sqrt(xp * xp + yp * yp)
        
        for (i, element) in enumerate(self._pol):
            if i == 0:
                zp = torch.full_like(r, element)
                tmp_r = r.clone()
            else:
                zp += element * tmp_r
                tmp_r *= r
        zp *= -1

        point3D = torch.stack([yp, xp, zp], dim = -1)
        point3D /= torch.linalg.norm(point3D, dim = -1, keepdim = True)
        return point3D

    def world2cam(self, point3D):
        assert point3D.shape[1] == 3
        
        point2D = torch.zeros_like(point3D[:, :2])

        norm = torch.linalg.norm(point3D[:, :2], dim = -1)
        valid_flag = (norm != 0)
        
        point2D[:, 0][~valid_flag] = self._yc
        point2D[:, 1][~valid_flag] = self._xc
        
        zero_flag = (point3D == 0).all(dim = 1)
        point2D[:, 0][zero_flag] = -1
        point2D[:, 1][zero_flag] = -1

        theta = -torch.arctan(point3D[:, 2][valid_flag] / norm[valid_flag])
        invnorm = 1.0 / norm[valid_flag]
        
        for (i, element) in enumerate(self._invpol):
            if i == 0:
                rho = torch.full_like(theta, element)
                tmp_theta = theta.clone()
            else:
                rho += element * tmp_theta
                tmp_theta *= theta

        u = point3D[:, 0][valid_flag] * invnorm * rho
        v = point3D[:, 1][valid_flag] * invnorm * rho
        point2D_valid_0 = v * self._affine[2] + u + self._yc
        point2D_valid_1 = v * self._affine[0] + u * self._affine[1] + self._xc

        if self._fov < 360:
            # finally deal with points are outside of fov
            thresh_theta = torch.deg2rad(torch.tensor(self._fov / 2)) - torch.pi / 2
            # set flag when  or point3D == (0, 0, 0)
            outside_flag = theta > thresh_theta
            point2D_valid_0[outside_flag] = -1
            point2D_valid_1[outside_flag] = -1

        point2D[:, 0][valid_flag] = point2D_valid_0
        point2D[:, 1][valid_flag] = point2D_valid_1

        return point2D

    @property
    def width(self):
        """ Getter for image width."""
        return self._img_size[1]

    @property
    def height(self):
        """ Getter for image height."""
        return self._img_size[0]

    @property
    def cx(self):
        """ Getter for image center cx (OpenCV format)."""
        return self._yc

    @property
    def cy(self):
        """ Getter for image center cy (OpenCV format)."""
        return self._xc

    def __repr__(self):
        print_list = []
        print_list.append(f"pol: {self._pol}")
        print_list.append(f"invpol: {self._invpol}")
        print_list.append(f"xc(col dir): {self._xc}, \tyc(row dir): {self._yc} in Ocam coord")
        print_list.append(f"affine: {self._affine}")
        print_list.append(f"img_size: {self._img_size}")
        if self._fov < 360:
            print_list.append(f"fov: {self._fov}")
        return "\n".join(print_list)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()
