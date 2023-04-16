use crate::game::*;
use crate::math::*;
use glutin::event::VirtualKeyCode;


impl Game {
    pub fn update_player(&mut self, dt: f32) {
        self.player_hp += self.player_hp_regen * dt;
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

        if self.lmb && self.t - self.player_t_shoot > self.player_cooldown {
            self.player_t_shoot = self.t;


            self.player_projectile_x.push(self.player_pos.x);
            self.player_projectile_y.push(self.player_pos.y);
            // let mut u = (self.mouse_pos - v2(0.5, 0.5)).normalize();
            // u.x /= (self.xres/self.yres) as f32;
            // u = u.normalize();
            let v = self.aim*self.player_proj_speed;
            self.player_projectile_vx.push(v.x);
            self.player_projectile_vy.push(v.y);
        }
    }

    pub fn draw_player(&mut self) {
        let p1 = v2(0.5, 0.0);
        let p2  = v2(0.75, 0.2);
        let p3  = v2(0.5, 0.4);
        let p4  = v2(0.25, 0.2);
    
        let p5 = p2.lerp(p3, 0.25);
        let p6 = p4.lerp(p3, 0.25);
        let p7 = p1.lerp(p3, 0.25);
    
        let p8 = v2(1.0, 0.5);
        let p9 = v2(0.5, 1.0);
        let p10 = v2(0.0, 0.5);
    
        let p11 = p7.lerp(p8, 0.5);
        let p12 = p7.lerp(p10, 0.5);
        let p13 = p7.lerp(p9, 0.5);
    
        let player_scale = 0.25;
        let player_transform = [
            player_scale, 0., self.player_pos.x - 0.5*player_scale,
            0., player_scale, self.player_pos.y - 0.5*player_scale,
            0., 0., 1.
        ];
    
        // self.world_geometry.put_rect(self.player_pos.rect_centered(player_s, player_s), v4(0., 0., 1., 1.), 0.7, v4(1., 1., 1., 1.), 0);
    
        let player_colour = v4(20.0, 0.5, 0.85, 1.0).hsv_to_rgb();
        let player_trim_colour = v4(180.0, 0.2, 0.7, 1.0).hsv_to_rgb();
        let player_face_colour = v4(0., 0., 0., 1.);
        let player_eye_colour = v4(1., 1., 1., 1.);
        let player_depth = 0.5;
    
        self.world_geometry.put_quad_transform(p1, v2(0., 0.), p2, v2(0., 0.), p3, v2(0., 0.), p4, v2(0., 0.), player_depth, player_trim_colour, 0, &player_transform);
        self.world_geometry.put_quad_transform(p7, v2(0., 0.), p5, v2(0., 0.), p3, v2(0., 0.), p6, v2(0., 0.), player_depth - 0.002, player_face_colour, 0, &player_transform);
        self.world_geometry.put_quad_transform(p7, v2(0., 0.), p8, v2(0., 0.), p9, v2(0., 0.), p10, v2(0., 0.), player_depth, player_colour, 0, &player_transform);
        self.world_geometry.put_quad_transform(p7, v2(0., 0.), p11, v2(0., 0.), p13, v2(0., 0.), p12, v2(0., 0.), player_depth - 0.001, player_trim_colour, 0, &player_transform);
    
        let p_leye = p4.lerp(p2, 0.35);
        let p_reye = p4.lerp(p2, 0.65); 
    
        let tv1 = v2(0.04, 0.0);
        let tv2 = v2(0.0, 0.04);
    
        self.world_geometry.put_triangle_transform(p_leye - tv1, v2(0.0, 0.0), p_leye + tv1, v2(0.0, 0.0), p_leye + tv2, v2(0.0, 0.0), player_depth - 0.003, player_eye_colour, 0, &player_transform);
        self.world_geometry.put_triangle_transform(p_reye - tv1, v2(0.0, 0.0), p_reye + tv1, v2(0.0, 0.0), p_reye + tv2, v2(0.0, 0.0), player_depth - 0.003, player_eye_colour, 0, &player_transform);
    
        self.world_geometry.put_rect_transform(v4(0.0, 1.0, 1.0, 0.25), v4(0., 0., 1., 1.,), player_depth + 0.01, v4(0., 0., 0., 1.0), 3, &player_transform);
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

            self.world_geometry.put_quad(p+o1, v2(0.0, 0.0), p+o2, v2(1.0, 0.0), p+o3, v2(1.0, 1.0), p+o4, v2(0.0, 1.0), 0.5, v4(1., 1., 0., 1.), 0);
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
                if vp.norm() < 0.2 {
                    kill = true;
                    self.enemy_hp[enemy_idx] -= self.player_damage;
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
