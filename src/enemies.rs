use crate::game::*;
use crate::math::*;
use crate::sound::*;

// draw flat so doesnt look all fucked up
// larger, more sinister eye


impl Game {
    pub fn draw_crystal_enemy(&mut self, x: f32, y: f32, colour: V4, scale: f32) {
        let tmat = [
            scale, 0., x,
            0., scale, y,
            0., 0., 1.
        ];

        let depth = 0.5;
        let w = 0.5;

        let p1 = v2(0.0, -1.0);
        let p2 = v2(w, -0.2);
        let p3 = v2(0.0, 1.0);
        let p4 = v2(-w, -0.2);

        let p5 = p1.lerp(p3, 0.2);
        let p6 = p1.lerp(p3, 0.6);

        self.world_geometry.put_triangle_transform(p1, v2(0., 0.), p2, v2(0., 0.), p5, v2(0., 0.), depth - 0.002, colour, 0, &tmat);
        self.world_geometry.put_triangle_transform(p1, v2(0., 0.), p5, v2(0., 0.), p4, v2(0., 0.), depth - 0.002, colour, 0, &tmat);
        
        self.world_geometry.put_triangle_transform(p6, v2(0., 0.), p2, v2(0., 0.), p3, v2(0., 0.), depth - 0.002, colour, 0, &tmat);
        self.world_geometry.put_triangle_transform(p4, v2(0., 0.), p6, v2(0., 0.), p3, v2(0., 0.), depth - 0.002, colour, 0, &tmat);
        
        self.world_geometry.put_quad_transform(p5, v2(0.0, 0.0), p2, v2(0.0, 0.0), p6, v2(0.0, 0.0), p4, v2(0.0, 0.0), depth - 0.001, v4(0., 0., 0., 1.), 0, &tmat);

        let eye_c = p5.lerp(p6, 0.5);
        let eye_v = (self.player_pos - v2(x, y)).normalize() * 0.11;

        let etvx = v2(0.1, 0.0);
        let etvy = v2(0.0, 0.3);

        self.world_geometry.put_quad_transform(eye_c + eye_v - etvy, v2(0.0, 0.0), eye_c + eye_v + etvx, v2(0.0, 0.0), eye_c + eye_v + etvy, v2(0.0, 0.0), eye_c + eye_v - etvx, v2(0., 0.), depth - 0.0015, v4(1., 0., 0., 1.), 0, &tmat);
        self.world_geometry.put_rect_transform(v4(-w, 1.0, 2.0*w, 0.25), v4(0., 0., 1., 1.,), depth + 0.01, v4(0., 0., 0., 1.0), 3, &tmat);



    }

    // 1 try at spawning enemies
    pub fn spawn_enemies(&mut self) {
        // self.enemies_spawn_seed = khash(self.enemies_spawn_seed);

        // let n = 4;
        // let nscale = 0.3;
        // let gw = LEVEL_W / n as f32;
        // let gh = LEVEL_H / n as f32;
        // for i in 0..n {
        //     for j in 0..n {
        //         let si = khash(self.enemies_spawn_seed + i * 1231234517 + j * 120312497 + 12312767);
        //         if krand(si * 1231237) < 0.8 {continue;}
        //         let oy = gh * krand(si * 12312377);
        //         let ox = gw * krand(si * 12301297);
        //         let x = i as f32 * gw + ox - LEVEL_W/2.0;
        //         let y = j as f32 * gh + oy - LEVEL_H/2.0;
        //         let richness = 1.7 * noise2(x * nscale, y * nscale, (self.level_seed * 12312397) as u32) - 0.3;
        //         let roll = krand(si);

        //         let score = richness * roll * (0.25*self.t_level + 10.0);

        //         if score > 2.0 {
        //             self.enemy_x.push(x);
        //             self.enemy_y.push(y);
        //             self.enemy_type.push(0);
        //             self.enemy_birth_t.push(self.t);
        //             self.enemy_attack_t.push(self.t);
        //             self.enemy_hp.push(3.5);
        //         }
        //     }
        // }
    }

    pub fn draw_enemies(&mut self) {
        for idx in 0..self.enemy_x.len() {
            if self.enemy_type[idx] == 0 {
                self.draw_crystal_enemy(self.enemy_x[idx], self.enemy_y[idx], v4(0., 0.6, 0., 1.), 0.25);
            }
        }
    }

    pub fn update_enemies(&mut self, dt: f32) {
        let enemy_r_shoot = 1.5;
        let enemy_r_engage = 3.0;
        let enemy_r_flee = 0.5;
        let enemy_speed = 0.1;
        let enemy_cooldown = 2.5;

        let projectile_speed = 1.0;

        let mut idx = self.enemy_x.len();
        while idx > 0 {
            idx -= 1;

            let mut enemy_vx = 0.0;
            let mut enemy_vy = 0.0;

            let v = self.player_pos - v2(self.enemy_x[idx], self.enemy_y[idx]);
            let vh = v.normalize();
            let r = v.norm();
            if r < enemy_r_flee {
                enemy_vx -= enemy_speed * vh.x;
                enemy_vy -= enemy_speed * vh.y;
            } else if r < enemy_r_engage {
                enemy_vx += enemy_speed * vh.x;
                enemy_vy += enemy_speed * vh.y;
            };
            self.enemy_x[idx] += enemy_vx*dt;
            self.enemy_y[idx] += enemy_vy*dt;

            
            if r < enemy_r_shoot && self.t - self.enemy_attack_t[idx] > enemy_cooldown {
                self.enemy_attack_t[idx] = self.t;
                self.enemy_projectile_x.push(self.enemy_x[idx]);
                self.enemy_projectile_y.push(self.enemy_y[idx]);
                self.enemy_projectile_vx.push(vh.x * projectile_speed);
                self.enemy_projectile_vy.push(vh.y * projectile_speed);
                self.enemy_projectile_type.push(self.enemy_type[idx]);
                self.prod.push(Sound { id: 2, birthtime: self.t, elapsed: 0.0, remaining: 0.3, magnitude: 0.08, mag_exp: 0.9999, frequency: 1.0, freq_exp: 1.0, wait: 0.0, phase: 0.0, samp: 0 }).unwrap();
            }

            if self.enemy_hp[idx] <= 0.0 {
                self.enemy_x.swap_remove(idx);
                self.enemy_y.swap_remove(idx);
                self.enemy_type.swap_remove(idx);
                self.enemy_birth_t.swap_remove(idx);
                self.enemy_attack_t.swap_remove(idx);
                self.enemy_hp.swap_remove(idx);
            }
        }
    }

    pub fn update_enemy_projectiles(&mut self, dt: f32) {
        let mut idx = self.enemy_projectile_x.len();
        while idx > 0 {
            idx -= 1;

            let vp = self.player_pos - v2(self.enemy_projectile_x[idx], self.enemy_projectile_y[idx]);

            let mut kill = false;

            self.enemy_projectile_x[idx] += self.enemy_projectile_vx[idx] * dt;
            self.enemy_projectile_y[idx] += self.enemy_projectile_vy[idx] * dt;

            if vp.norm() < 0.2 {
                self.prod.push(Sound { id: 1, birthtime: self.t, elapsed: 0.0, remaining: 0.4, magnitude: 0.1, mag_exp: 0.9999, frequency: 130.0, freq_exp: 1.0, wait: 0.0, phase: 0.0, samp: 0 }).unwrap();
                self.prod.push(Sound { id: 1, birthtime: self.t, elapsed: 0.0, remaining: 0.4, magnitude: 0.1, mag_exp: 0.9999, frequency: 140.0, freq_exp: 1.0, wait: 0.0, phase: 0.0, samp: 0 }).unwrap();
                self.player_hp -= 0.2;
                kill = true;
            }
            
            if self.enemy_projectile_x[idx] < -LEVEL_W/2.0 || self.enemy_projectile_x[idx] > LEVEL_W/2.0 || self.enemy_projectile_y[idx] < -LEVEL_H/2.0 || self.enemy_projectile_y[idx] > LEVEL_H/2.0 {
                kill = true;
            }

            if kill {
                self.enemy_projectile_type.swap_remove(idx);
                self.enemy_projectile_x.swap_remove(idx);
                self.enemy_projectile_vx.swap_remove(idx);
                self.enemy_projectile_y.swap_remove(idx);
                self.enemy_projectile_vy.swap_remove(idx);
            }
        }
    }

    pub fn draw_enemy_projectiles(&mut self) {
        let projectile_s = 0.1;

        for idx in 0..self.enemy_projectile_x.len() {
            let p = v2(self.enemy_projectile_x[idx], self.enemy_projectile_y[idx]);
            let v = v2(self.enemy_projectile_vx[idx], self.enemy_projectile_vy[idx]);
            let u = v.normalize();
            let orth = v2(-u.y, u.x);

            let o1 = (-orth+2.0*u)*projectile_s;
            let o2 = (orth+2.0*u)*projectile_s;
            let o3 = (orth-2.0*u)*projectile_s;
            let o4 = (-orth-2.0*u)*projectile_s;

            self.world_geometry.put_quad(p+o1, v2(0.0, 0.0), p+o2, v2(1.0, 0.0), p+o3, v2(1.0, 1.0), p+o4, v2(0.0, 1.0), 0.35, v4(1., 0.7, 0., 0.9), 5);
        }
    }

    
}