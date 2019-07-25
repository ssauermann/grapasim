#include "Hilbert3D.h"

#define back _back();
#define front _front();
#define up _up();
#define down _down();
#define left _left();
#define right _right();

#define xyz _xyz(depth - 1);
#define xYZ _xYZ(depth - 1);
#define yzx _yzx(depth - 1);
#define yZX _yZX(depth - 1);
#define zxy _zxy(depth - 1);
#define zXY _zXY(depth - 1);
#define XyZ _XyZ(depth - 1);
#define XYz _XYz(depth - 1);
#define YzX _YzX(depth - 1);
#define YZx _YZx(depth - 1);
#define ZxY _ZxY(depth - 1);
#define ZXy _ZXy(depth - 1);

void Hilbert3D::_xyz(int depth) {
    if (depth == 0) {
        execute();
    } else {
        yzx
        back
        zxy
        right
        zxy
        front
        XYz
        up
        XYz
        back
        ZxY
        left
        ZxY
        front
        yZX
    }
}

void Hilbert3D::_xYZ(int depth) {
    if (depth == 0) {
        execute();
    } else {
        yZX
        front
        zXY
        right
        zXY
        back
        XyZ
        down
        XyZ
        front
        ZXy
        left
        ZXy
        back
        yzx
    }
}

void Hilbert3D::_yzx(int depth) {
    if (depth == 0) {
        execute();
    } else {
        zxy
        right
        xyz
        up
        xyz
        left
        YzX
        back
        YzX
        right
        xYZ
        down
        xYZ
        left
        ZXy
    }
}

void Hilbert3D::_yZX(int depth) {
    if (depth == 0) {
        execute();
    } else {
        zXY
        right
        xYZ
        down
        xYZ
        left
        YZx
        front
        YZx
        right
        xyz
        up
        xyz
        left
        ZxY
    }
}

void Hilbert3D::_zxy(int depth) {
    if (depth == 0) {
        execute();
    } else {
        xyz
        up
        yzx
        back
        yzx
        down
        zXY
        right
        zXY
        up
        YZx
        front
        YZx
        down
        XyZ
    }
}

void Hilbert3D::_zXY(int depth) {
    if (depth == 0) {
        execute();
    } else {
        xYZ
        down
        yZX
        front
        yZX
        up
        zxy
        right
        zxy
        down
        YzX
        back
        YzX
        up
        XYz
    }
}

void Hilbert3D::_XyZ(int depth) {
    if (depth == 0) {
        execute();
    } else {
        YzX
        back
        ZxY
        left
        ZxY
        front
        xYZ
        down
        xYZ
        back
        zxy
        right
        zxy
        front
        YZx
    }
}

void Hilbert3D::_XYz(int depth) {
    if (depth == 0) {
        execute();
    } else {
        YZx
        front
        ZXy
        left
        ZXy
        back
        xyz
        up
        xyz
        front
        zXY
        right
        zXY
        back
        YzX
    }
}

void Hilbert3D::_YzX(int depth) {
    if (depth == 0) {
        execute();
    } else {
        ZxY
        left
        XyZ
        down
        XyZ
        right
        yzx
        back
        yzx
        left
        XYz
        up
        XYz
        right
        zXY
    }
}

void Hilbert3D::_YZx(int depth) {
    if (depth == 0) {
        execute();
    } else {
        ZXy
        left
        XYz
        up
        XYz
        right
        yZX
        front
        yZX
        left
        XyZ
        down
        XyZ
        right
        zxy
    }
}

void Hilbert3D::_ZxY(int depth) {
    if (depth == 0) {
        execute();
    } else {
        XyZ
        down
        YzX
        back
        YzX
        up
        ZXy
        left
        ZXy
        down
        yZX
        front
        yZX
        up
        xyz
    }
}

void Hilbert3D::_ZXy(int depth) {
    if (depth == 0) {
        execute();
    } else {
        XYz
        up
        YZx
        front
        YZx
        down
        ZxY
        left
        ZxY
        up
        yzx
        back
        yzx
        down
        xYZ
    }
}
