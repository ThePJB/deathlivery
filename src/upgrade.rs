use crate::game::*;
use crate::math::*;
use crate::sound::fail_sound;

// yea i should just make it like int levels and do a to float interpretation

#[derive(Clone)]
pub struct Upgrade {
    add: [f32; 8],
    mul: [f32; 8],
    name: String,
    rarity: usize,
}

const STATS_NONE: [f32; 8] = [0., 0., 0., 0., 0., 0., 0., 0.];
const MULT_ONE: [f32; 8] = [1., 1., 1., 1., 1., 1., 1., 1.];

pub fn upgrade_common() -> Vec<Upgrade> {
    vec![
        Upgrade {
            add: [-100.0, 0.0, 0.2, 0.01, 0., 0., 0., 0.],
            mul: MULT_ONE,
            name: "healthy".to_string(),
            rarity: 0,
        },
        Upgrade {
            add: [-120.0, 0., 0., 0., 0.1, 0.0, 0., 0.],
            mul: MULT_ONE,
            name: "zoomy".to_string(),
            rarity: 0,
        },
        Upgrade {
            add: [-150.0, 0.25, 0., 0., 0.0, 0.0, 0., 0.],
            mul: MULT_ONE,
            name: "damaging".to_string(),
            rarity: 0,
        },
        Upgrade {
            add: [-110.0, 0.0, 0., 0., 0.0, 0.0, 0., 0.],
            mul: [1., 1., 1., 1., 1., 1., 1., 0.9],
            name: "delay".to_string(),
            rarity: 0,
        },
        Upgrade {
            add: [-80.0, 0.0, 0., 0., 0.0, 0.0, 0.2, 0.],
            mul: MULT_ONE,
            name: "projspeed".to_string(),
            rarity: 0,
        },
        Upgrade {
            add: [-150., 0., 0., 0., 0., 0.15, 0., 0.],
            mul: MULT_ONE,
            name: "succ".to_string(),
            rarity: 0,
        }
    ]
}

pub fn upgrade_uncommon() -> Vec<Upgrade> {
    vec![
        Upgrade {
            add: [-300.0, 0.0, 0.0, 0.00, 0., 0., 0., 0.],
            mul: [1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            name: "rapidfire".to_string(),
            rarity: 1,
        },
        Upgrade {
            add: [-200.0, 0.0, 0.0, 0.00, 0., 0., 0., 0.],
            mul: [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.9, 1.5],
            name: "powerful".to_string(),
            rarity: 1,
        },
    ]
}

pub fn upgrade_rare() -> Vec<Upgrade> {
    vec![
        Upgrade {
            add: [-500.0, 0.00, 0.01, 0.00, 0., 0., 0., 0.],
            mul: [1.0, 3.0, 0.00, 1., 1., 1., 1., 1.],
            name: "samurai".to_string(),
            rarity: 2,
        },
        Upgrade {
            add: [-500.0, 0.00, 0.00, 0.00, 0.4, 0.2, 0., 0.],
            mul: [1.0, 0.0, 1., 1., 1., 1., 1., 1.],
            name: "pacifist".to_string(),
            rarity: 2,
        },
    ]
}

impl Game {
    pub fn roll_upgrade(&self, seed: usize) -> Upgrade {
        let roll = khash(seed) % (self.upgrade_common.len() + 1);
        if roll >= self.upgrade_common.len() {
            let roll = khash(seed) % (self.upgrade_uncommon.len());
            if roll >= self.upgrade_uncommon.len() {
                let roll = khash(seed) % (self.upgrade_rare.len() - 1);
                return self.upgrade_rare[roll].clone();
            }
            return self.upgrade_uncommon[roll].clone();
        }
        return self.upgrade_common[roll].clone();
    }
    pub fn disp_upgrade(&mut self, mut x: f32, mut y: f32, cw: f32, ch: f32, depth: f32, upgrade: Upgrade) {
        let tileset = ['A', 'B', 'C', 'D', 'E', 'F','G', 'H'];
        self.screen_geometry_unscaled.put_string_left(&upgrade.name, x, y, cw, ch, depth, v4(1., 1., 1., 1.));
        x += upgrade.name.len() as f32 * cw;
        for i in 0..8 {
            if upgrade.mul[i] == 1.0 && upgrade.add[i] == 0.0 {continue;}
            x += cw;
            self.screen_geometry_unscaled.put_glyph(tileset[i], v4(x, y, cw, ch), depth, v4(1., 1., 1., 1.)); x += cw;
            if upgrade.mul[i] != 1.0 {
                let s = format!("x{}", upgrade.mul[i]);
                self.screen_geometry_unscaled.put_string_left(&s, x, y, cw, ch, depth, v4(1., 1., 1., 1.)); x += s.len() as f32 * cw;
            }
            if upgrade.add[i] != 0.0 {
                let s = format!("{}", upgrade.add[i]);
                self.screen_geometry_unscaled.put_string_left(&s, x, y, cw, ch, depth, v4(1., 1., 1., 1.)); x += s.len() as f32 * cw;
            }
        }
    }
    pub fn handle_upgrade(&mut self, selection: usize) {
        if !self.in_portal {return;}
        let u = self.roll_upgrade(self.upgrade_seed[selection]);
        if (self.gems as i64 + u.add[0] as i64) < 0 {
            println!("too poor");
            self.prod.push(fail_sound(self.t)).unwrap();
            return;
        }
        self.upgrade_seed[selection] = 123179273 * khash(self.upgrade_seed[selection] + 1231237 * 12312231667) + 1231294147;
        for i in 0..8 {
            self.do_stat_apply(i, u.mul[i], u.add[i]);
        }

        // otherwise apply stat changes
        // update seed
    }
    pub fn do_stat_apply(&mut self, stat: usize, mul: f32, add: f32) {
        match stat {
            0 => {
                self.gems = ((self.gems as f32 * mul) + add).round() as usize;
            },
            1 => {
                self.player_damage = self.player_damage * mul + add;
            },
            2 => {
                self.player_hp_max = self.player_hp_max * mul + add;
            },
            3 => {
                self.player_hp_regen = self.player_hp_regen * mul + add;
            },
            4 => {
                self.player_speed = self.player_speed * mul + add;
            },
            5 => {
                self.player_succ = self.player_succ * mul + add;
            },
            6 => {
                self.player_proj_speed = self.player_proj_speed * mul + add;
            },
            7 => {
                self.player_cooldown = self.player_cooldown * mul + add;
            },
            _ => {},
        }
    }
}