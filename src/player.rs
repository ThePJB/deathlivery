use crate::game::*;
use crate::math::*;
use crate::sound::*;
use glutin::event::VirtualKeyCode;


impl Game {
    pub fn update_player(&mut self, dt: f32) {
        self.player_hp += self.player_hp_regen * dt;
        self.player_hp = self.player_hp.min(self.player_hp_max);
        self.player_vel = v2(0., 0.);
        if self.held_keys.contains(&VirtualKeyCode::A) {
            self.player_vel.x -= self.player_speed;
        }
        if self.held_keys.contains(&VirtualKeyCode::D) {
            self.player_vel.x += self.player_speed;
        }
        if self.held_keys.contains(&VirtualKeyCode::W) {
            self.player_vel.y -= self.player_speed;
        }
        if self.held_keys.contains(&VirtualKeyCode::S) {
            self.player_vel.y += self.player_speed;
        }
        self.player_pos = self.player_pos + self.player_vel * dt;

        if self.lmb {
            if self.t - self.player_t_shoot > self.player_cooldown {
                self.player_t_shoot = self.t;


                self.player_projectile_x.push(self.player_pos.x);
                self.player_projectile_y.push(self.player_pos.y);
                // let mut u = (self.mouse_pos - v2(0.5, 0.5)).normalize();
                // u.x /= (self.xres/self.yres) as f32;
                // u = u.normalize();
                let v = self.aim*self.player_proj_speed;
                self.player_projectile_vx.push(v.x);
                self.player_projectile_vy.push(v.y);
                self.prod.push(Sound { id: 2, birthtime: self.t, elapsed: 0.0, remaining: 0.2, magnitude: 0.1, mag_exp: 0.9995, frequency: 1.0, freq_exp: 1.0, wait: 0.0, phase: 0.0, samp: 0 }).unwrap();
            } else {
                if self.lmb_this_frame {
                    self.prod.push(Sound { id: 1, birthtime: self.t, elapsed: 0.0, remaining: 0.2, magnitude: 0.2, mag_exp: 0.9995, frequency: 110.0, freq_exp: 1.0, wait: 0.0, phase: 0.0, samp: 0 }).unwrap();
                }
            }
        }
    }

    pub fn draw_player(&mut self, x: f32, y: f32, scale: f32, theta: f32, shadow: bool) {
        let b1 = v2(0.0, -1.0);
        let b2 = v2(1.0, 0.0);
        let b3 = v2(0.0, 1.0);
        let b4 = v2(-1.0, 0.0);

        let hu = v2(0.5, 0.0);
        let hv = v2(0.0, 0.5);

        let h1 = b1 - hv;
        let h2 = b1 + hu;
        let h3 = b1 + hv;
        let h4 = b1 - hu;

        let f1 = b1;
        let f2 = b1.lerp(b2, 0.25);
        let f3 = b1.lerp(b3, 0.25);
        let f4 = b1.lerp(b4, 0.25);

        let s1 = b1;
        let s2 = b1.lerp(b2, 0.5);
        let s3 = b1.lerp(b3, 0.5);
        let s4 = b1.lerp(b4, 0.5);

        // no transform just calculate off verlet points
        // let player_x = 0.0; // make avg of players
        // let player_transform = mat_srt(self.player_pos.x, self.player_pos.y, scale, theta);

        let body_colour = v4(20.0, 0.5, 0.85, 1.0).hsv_to_rgb();
        let scarf_colour = v4(180.0, 0.2, 0.7, 1.0).hsv_to_rgb();
        let face_colour = v4(0., 0., 0., 1.);
        let eye_colour = v4(1., 1., 1., 1.);
        let player_depth = 0.5;
    
        // self.world_geometry.put_quad_transform(b1, v2(0., 0.), b2, v2(0., 0.), b3, v2(0., 0.), b4, v2(0., 0.), player_depth, body_colour, 0, &player_transform);
        // self.world_geometry.put_quad_transform(s1, v2(0., 0.), s2, v2(0., 0.), s3, v2(0., 0.), s4, v2(0., 0.), player_depth, scarf_colour, 0, &player_transform);
        // self.world_geometry.put_quad_transform(h1, v2(0., 0.), h2, v2(0., 0.), h3, v2(0., 0.), h4, v2(0., 0.), player_depth, scarf_colour, 0, &player_transform);
        // self.world_geometry.put_quad_transform(f1, v2(0., 0.), f2, v2(0., 0.), f3, v2(0., 0.), f4, v2(0., 0.), player_depth, face_colour, 0, &player_transform);
    
        let p_leye = f4.lerp(f2, 0.35);
        let p_reye = f4.lerp(f2, 0.65); 
    
        let tv1 = v2(0.04, 0.0);
        let tv2 = v2(0.0, 0.04);
    
        self.world_geometry.put_triangle_transform(p_leye - tv1, v2(0.0, 0.0), p_leye + tv1, v2(0.0, 0.0), p_leye + tv2, v2(0.0, 0.0), player_depth - 0.003, eye_colour, 0, &player_transform);
        self.world_geometry.put_triangle_transform(p_reye - tv1, v2(0.0, 0.0), p_reye + tv1, v2(0.0, 0.0), p_reye + tv2, v2(0.0, 0.0), player_depth - 0.003, eye_colour, 0, &player_transform);
    
        if shadow {
            self.world_geometry.put_rect_transform(v4(-1.0, 1.0, 2.0, 0.5), v4(0., 0., 1., 1.,), player_depth + 0.01, v4(0., 0., 0., 1.0), 3, &player_transform);
        }
    }

    pub fn draw_player_projectiles(&mut self) {
        let projectile_s = 0.05;

        for idx in 0..self.player_projectile_x.len() {
            let p = v2(self.player_projectile_x[idx], self.player_projectile_y[idx]);
            let v = v2(self.player_projectile_vx[idx], self.player_projectile_vy[idx]);
            let u = v.normalize();
            let orth = v2(-u.y, u.x);

            let o1 = (-orth+2.0*u)*projectile_s;
            let o2 = (orth+2.0*u)*projectile_s;
            let o3 = (orth-2.0*u)*projectile_s;
            let o4 = (-orth-2.0*u)*projectile_s;

            self.world_geometry.put_quad(p+o1, v2(0.0, 0.0), p+o2, v2(1.0, 0.0), p+o3, v2(1.0, 1.0), p+o4, v2(0.0, 1.0), 0.35, v4(0., 0.7, 1., 0.9), 5);
        }
    }
    
    pub fn update_player_projectiles(&mut self, dt: f32) {
        let mut idx = self.player_projectile_x.len();
        while idx > 0 {
            idx -= 1;


            let mut kill = false;

            self.player_projectile_x[idx] += self.player_projectile_vx[idx] * dt;
            self.player_projectile_y[idx] += self.player_projectile_vy[idx] * dt;

            for enemy_idx in 0..self.enemy_x.len() {
                let ep = v2(self.enemy_x[enemy_idx], self.enemy_y[enemy_idx]);
                let vp = ep - v2(self.player_projectile_x[idx], self.player_projectile_y[idx]);
                if vp.norm() < 0.3 {
                    kill = true;
                    // self.enemy_hp[enemy_idx] -= self.player_damage; // how much damage
                    self.prod.push(Sound { id: 1, birthtime: self.t, elapsed: 0.0, remaining: 0.2, magnitude: 0.1, mag_exp: 0.9995, frequency: 660.0 * 5./6. , freq_exp: 1.0, wait: 0.0, phase: 0.0, samp: 0 }).unwrap();
                }
            }

            
            if self.player_projectile_x[idx] < -LEVEL_W/2.0 || self.player_projectile_x[idx] > LEVEL_W/2.0 || self.player_projectile_y[idx] < -LEVEL_H/2.0 || self.player_projectile_y[idx] > LEVEL_H/2.0 {
                kill = true;
            }

            if kill {
                self.player_projectile_x.swap_remove(idx);
                self.player_projectile_vx.swap_remove(idx);
                self.player_projectile_y.swap_remove(idx);
                self.player_projectile_vy.swap_remove(idx);
            }
        }
    }
}
