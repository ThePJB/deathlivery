use crate::math::*;
use crate::kimg::*;

// polynomial smooth min
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k-(a-b).abs()).max(0.0 )/k;
    return a.min(b) - h*h*k*(1.0/4.0);
}


// unless it was not only SD but SD + vector field to nearest point
// i hope its preserved by domain warp
pub fn sd_circle(p: V2, r: f32) -> f32 {
    p.norm() - r
}

pub fn op_sub(d1: f32, d2: f32) -> f32 {
    (-d1).max(d2)
}

pub fn sd_terrain(p: V2) -> f32 {
    let sdc1 = sd_circle(p - v2(0.0, -0.5), 0.33);
    let sdc2 = sd_circle(p - v2(0.35, 0.5), 0.4);
    let sdc3 = sd_circle(p - v2(0.15, -0.15), 0.25);

    let k = 0.1;
    let d_circles = smin(smin(sdc1, sdc2, k), sdc3, k);

    let d_box = sd_box(p, v2(1.0, 1.0));

    // op_sub(d_box, d_circles)

    // d_box
    d_circles


}

pub fn sd_box(p: V2, b: V2) -> f32 {
    let d = v2(p.x.abs(), p.y.abs()) - b;
    let q = v2(d.x.max(0.0), d.y.max(0.0));
    q.norm() + (d.x.max(d.y).min(0.0))
}


#[test]
fn test_terrain() {
    use crate::kimg::ImageBuffer;

    let xres = 400;
    let yres = 400;

    let mut im = ImageBuffer::new(xres, yres);
    for i in 0..xres {
        for j in 0..yres {
            let x = (i as f32 - (xres/2) as f32) * 2.0 / xres as f32;
            let y = (j as f32 - (yres/2) as f32) * 2.0 / yres as f32;

            let p = v2(x,y);

            if sd_terrain(p) > 0.0 {
                im.set_px(i, j, (255, 255, 255));
            } else {
                im.set_px(i, j, (0, 0, 0));
            }
        }
    }
    im.dump_to_file("level.png");
}