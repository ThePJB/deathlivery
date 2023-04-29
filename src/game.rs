use glow::HasContext;
use glow::*;
use glutin::event::VirtualKeyCode;
use glutin::event::Event;
use glutin::event::WindowEvent;
use glutin::event::MouseButton;
use glutin::event::ElementState;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use crate::math::*;
use crate::vertex_buffer::*;
use std::collections::HashSet;
pub use std::f32::consts::PI;
use cpal::traits::*;
use ringbuf::*;

use crate::player;
use crate::upgrade::*;
use crate::sound::*;


pub const LEVEL_W: f32 = 20.0;
pub const LEVEL_H: f32 = 20.0;

pub const FNTH: f32 = 0.1;
pub const FNTW: f32 = FNTH * 12./16.;

pub struct Game {
    pub xres: i32,
    pub yres: i32,
    pub window: glutin::ContextWrapper<glutin::PossiblyCurrent, glutin::window::Window>,
    pub gl: glow::Context,
    pub stream: cpal::Stream,
    pub prod: Producer<Sound>,
    
    pub mouse_pos: V2,
    pub aim: V2,
    pub lmb: bool,
    pub lmb_this_frame: bool,
    pub held_keys: HashSet<VirtualKeyCode>,

    pub t: f32,
    pub t_last: Instant,
    pub t_level: f32,

    pub program: glow::NativeProgram,
    pub vao: glow::NativeVertexArray,
    pub vbo: glow::NativeBuffer,
    pub texture: glow::NativeTexture,

    pub screen_geometry: VertexBuffer,
    pub screen_geometry_unscaled: VertexBuffer,
    pub world_geometry: VertexBuffer,

    pub in_portal: bool,

    pub player_pos: V2,
    pub player_vel: V2,
    pub player_succ: f32,
    pub player_hp: f32,
    pub player_hp_max: f32,
    pub player_hp_regen: f32,
    pub player_speed: f32,
    pub player_t_shoot: f32,
    pub player_cooldown: f32,
    pub player_damage: f32,
    pub player_proj_speed: f32,

    pub warp_price: usize,
    pub reroll_price: usize,
    pub upgrade_seed: [usize; 3],

    pub upgrade_common: Vec<Upgrade>,
    pub upgrade_uncommon: Vec<Upgrade>,
    pub upgrade_rare: Vec<Upgrade>,

    pub seed: usize,
    pub level_seed: usize,
    pub enemies_spawn_seed: usize,

    pub gem_x: Vec<f32>,
    pub gem_y: Vec<f32>,
    pub gem_vx: Vec<f32>,
    pub gem_vy: Vec<f32>,
    pub gem_type: Vec<usize>,

    pub gems: usize,

    pub spawn_enemies_counter: f32,

    pub enemy_x: Vec<f32>,
    pub enemy_y: Vec<f32>,
    pub enemy_type: Vec<usize>,
    pub enemy_birth_t: Vec<f32>,
    pub enemy_attack_t: Vec<f32>,
    pub enemy_hp: Vec<f32>,
    
    pub enemy_projectile_x: Vec<f32>,
    pub enemy_projectile_y: Vec<f32>,
    pub enemy_projectile_vx: Vec<f32>,
    pub enemy_projectile_vy: Vec<f32>,
    pub enemy_projectile_type: Vec<usize>,
    
    pub player_projectile_x: Vec<f32>,
    pub player_projectile_y: Vec<f32>,
    pub player_projectile_vx: Vec<f32>,
    pub player_projectile_vy: Vec<f32>,

}

impl Game {
    pub unsafe fn new(event_loop: &glutin::event_loop::EventLoop<()>) -> Game {
        // ====================
        // Sound Init
        // ====================
        let rb = RingBuffer::<Sound>::new(200);
        let (mut prod, mut cons) = rb.split();
        let stream = stream_setup_for(sample_next, cons).expect("no can make stream");
        stream.play().expect("no can play stream");


        let xres = 800i32;
        let yres = 800i32;
        let window_builder = glutin::window::WindowBuilder::new()
            .with_title("Deathlivery")
            .with_inner_size(glutin::dpi::PhysicalSize::new(xres, yres));
        let window = glutin::ContextBuilder::new()
            .with_vsync(true)
            .build_windowed(window_builder, &event_loop)
            .unwrap()
            .make_current()
            .unwrap();
        let gl = glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _);
        gl.enable(DEPTH_TEST);
        // gl.enable(CULL_FACE);
        gl.blend_func(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
        gl.depth_func(LEQUAL);
        gl.enable(BLEND);
        // gl.debug_message_callback(|a, b, c, d, msg| {
        //     println!("{} {} {} {} msg: {}", a, b, c, d, msg);
        // });

        let vbo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(vao));
        
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 0);
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(1, 4, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 4*3);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 4*3 + 4*4);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_i32(3, 1, glow::UNSIGNED_INT, 4*4 + 4*3 + 4*2 + 4, 4*3 + 4*4 + 4*2);
        gl.enable_vertex_attrib_array(3);


        // Shader
        let program = gl.create_program().expect("Cannot create program");
    
        let vs = gl.create_shader(glow::VERTEX_SHADER).expect("cannot create vertex shader");
        gl.shader_source(vs, include_str!("shader.vert"));
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            panic!("{}", gl.get_shader_info_log(vs));
        }
        gl.attach_shader(program, vs);

        let fs = gl.create_shader(glow::FRAGMENT_SHADER).expect("cannot create fragment shader");
        gl.shader_source(fs, include_str!("shader.frag"));
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            panic!("{}", gl.get_shader_info_log(fs));
        }
        gl.attach_shader(program, fs);

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{}", gl.get_program_info_log(program));
        }
        gl.detach_shader(program, fs);
        gl.delete_shader(fs);
        gl.detach_shader(program, vs);
        gl.delete_shader(vs);

        let png_bytes = include_bytes!("../tex.png").as_ref();
        let decoder = png::Decoder::new(png_bytes);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let bytes = &buf[..info.buffer_size()];

        let texture = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D, 
            0, 
            glow::RGBA as i32, 
            info.width as i32, info.height as i32, 
            0, 
            RGBA, 
            glow::UNSIGNED_BYTE, 
            Some(bytes)
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

        gl.generate_mipmap(glow::TEXTURE_2D);

        let initial_seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as usize;

        Game {
            program,
            vao,
            vbo,
            texture,
            xres,
            yres,
            window,
            gl,
            screen_geometry: VertexBuffer::default(),
            screen_geometry_unscaled: VertexBuffer::default(),
            world_geometry: VertexBuffer::default(),
            stream,
            prod,
            held_keys: HashSet::new(),
            mouse_pos: v2(0., 0.),
            aim: v2(0., 0.),
            lmb: false,
            lmb_this_frame: false,
            t_last: Instant::now(),
            t: 0.0,
            t_level: 0.0,
            in_portal: false,
            player_pos: v2(0., 0.),
            player_vel: v2(0., 0.),
            player_succ: 0.5,
            player_hp: 1.0,
            player_hp_max: 1.0,
            player_hp_regen: 0.01,
            player_speed: 0.5,
            player_t_shoot: -999.0, 
            player_cooldown: 0.7,
            player_damage: 0.5,
            player_proj_speed: 1.5,
            warp_price: 100,
            reroll_price: 10,
            seed: initial_seed,
            enemies_spawn_seed: khash(initial_seed * 1231247),
            level_seed: khash(initial_seed * 129371237),
            upgrade_seed: [khash(initial_seed * 123237), khash(initial_seed * 9123912397), khash(initial_seed * 231241257)],
            gem_type: vec![],
            gem_x: vec![],
            gem_y: vec![],
            gem_vx: vec![],
            gem_vy: vec![],
            gems: 0,
            spawn_enemies_counter: 0.0,
            enemy_x: vec![],
            enemy_y: vec![],
            enemy_type: vec![],
            enemy_birth_t: vec![],
            enemy_attack_t: vec![],
            enemy_hp: vec![],
            enemy_projectile_x: vec![],
            enemy_projectile_y: vec![],
            enemy_projectile_vx: vec![],
            enemy_projectile_vy: vec![],
            enemy_projectile_type: vec![],
            player_projectile_x: vec![],
            player_projectile_y: vec![],
            player_projectile_vx: vec![],
            player_projectile_vy: vec![],
            upgrade_common: upgrade_common(),
            upgrade_uncommon: upgrade_uncommon(),
            upgrade_rare: upgrade_rare(),
        }
    }

    pub unsafe fn handle_event(&mut self, event: Event<()>) {
        match event {
            Event::LoopDestroyed |
            Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => {
                std::process::exit(0);
            }

            Event::WindowEvent {event, .. } => {
                match event {
                    WindowEvent::CursorMoved {position, .. } => {
                        self.mouse_pos.x = position.x as f32 / self.xres as f32;
                        self.mouse_pos.y = position.y as f32 / self.yres as f32;
                        self.aim.x = (position.x as f32 - self.xres as f32/2.0) as f32 / self.xres as f32;
                        self.aim.y = (position.y as f32 - self.yres as f32/2.0) as f32 / self.yres as f32;
                        self.aim.x *= (self.xres as f32 / self.yres as f32);
                        self.aim = self.aim.normalize();
                    },
                    WindowEvent::Resized(size) => {
                        self.xres = size.width as i32;
                        self.yres = size.height as i32;
                        self.gl.viewport(0, 0, size.width as i32, size.height as i32)
                    },
                    WindowEvent::MouseInput {state: ElementState::Pressed, button: MouseButton::Left, ..} => {
                        self.lmb = true;
                        self.lmb_this_frame = true;
                    },
                    WindowEvent::MouseInput {state: ElementState::Released, button: MouseButton::Left, ..} => {
                        self.lmb = false;
                    },
                    WindowEvent::KeyboardInput {input, ..} => {
                        match input {
                            glutin::event::KeyboardInput {virtual_keycode: Some(code), state: ElementState::Pressed, ..} => {
                                self.held_keys.insert(code);
                                if self.player_hp < 0.0 {
                                    self.restart();
                                }
                            },
                            glutin::event::KeyboardInput {virtual_keycode: Some(code), state: ElementState::Released, ..} => {
                                self.held_keys.remove(&code);
                                match code {
                                    VirtualKeyCode::Escape => {
                                    },
                                    VirtualKeyCode::Key1 => self.handle_upgrade(0),
                                    VirtualKeyCode::Key2 => self.handle_upgrade(1),
                                    VirtualKeyCode::Key3 => self.handle_upgrade(2),
                                    VirtualKeyCode::R => {
                                        if self.in_portal {
                                            if self.gems >= self.reroll_price {
                                                self.gems -= self.reroll_price;
                                                self.upgrade_seed[0] = khash(self.upgrade_seed[0] * 12312541247);
                                                self.upgrade_seed[1] = khash(self.upgrade_seed[1] * 1231377);
                                                self.upgrade_seed[2] = khash(self.upgrade_seed[2] * 5438373737);
                                            } else {
                                                self.prod.push(fail_sound(self.t)).unwrap();
                                            }
                                        }
                                    },
                                    VirtualKeyCode::T => {
                                        if self.in_portal {
                                            self.in_portal = false;
                                            // regenerate level
                                            self.t_level = 0.0;
                                            // regen gems etc
                                            self.level_seed = khash(self.level_seed * 12412317);
                                            self.populate_with_gems();
                                        } else {
                                            if self.gems >= self.warp_price {
                                                self.gems -= self.warp_price;
                                                self.in_portal = true;
                                                self.zero_state();
                                                self.player_pos = v2(0., 0.);
                                            }
                                        }
                                    },
                                    _ => {},
                                }
                            },
                            _ => {},
                        }
                    },
                    _ => {},
                }
            },
            Event::MainEventsCleared => self.frame(),
            _ => {},
        }
    }

    pub unsafe fn frame(&mut self) {
        let t_now = Instant::now();
        let dt = (t_now - self.t_last).as_secs_f32();
        self.t += dt;
        self.t_last = t_now;

        self.screen_geometry = VertexBuffer::default();
        self.screen_geometry_unscaled = VertexBuffer::default();
        self.world_geometry = VertexBuffer::default();

        let aspect = self.xres as f32 / self.yres as f32;

        if !self.in_portal && self.player_hp > 0.0 {
            self.simulate(dt);
        }

        let cam_x = self.player_pos.x;
        let cam_y = self.player_pos.y;
        let scale = 1.0;
        let x_scale = scale/aspect;   // either this or 1/aspect
        let y_scale = scale;

        if self.player_hp <= 0.0 {
            println!("you are dead");
            self.screen_geometry_unscaled.put_string_centered("-- you have died --",  0.0, 0.0, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.));
            self.screen_geometry_unscaled.put_string_centered("press any key to restart",  0.0, FNTH, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.));

        }

        if self.in_portal {
            let mut y = -1.0 + FNTH;
            self.screen_geometry_unscaled.put_rect(v4(-1., y, FNTW/aspect* 8.0, FNTH*8.0), v4(0., 0., 1., 1.), 0.8, v4(0.5, 0.5, 0.5, 0.8), 0);
            self.screen_geometry_unscaled.put_rect(v4(-1., -1., 2., 2.), v4(0., 0., 1., 1.,), 0.9, v4(0., 0., 0., 1.), 0);
            self.screen_geometry_unscaled.put_string_left(&format!("A{}", self.gems),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("B{}", self.player_damage),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("C{}", self.player_hp_max),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("D{}", self.player_hp_regen),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("E{}", self.player_speed),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("F{}", self.player_succ),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("G{}", self.player_proj_speed),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("H{}", self.player_cooldown),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;

            y += FNTH;
            y += FNTH;
            y += FNTH;
            
            self.screen_geometry_unscaled.put_rect(v4(-1., y, FNTW/aspect* 8.0, FNTH*2.0), v4(0., 0., 1., 1.), 0.8, v4(0.5, 0.5, 0.5, 0.8), 0);
            self.screen_geometry_unscaled.put_string_left("Z[r]",  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            self.screen_geometry_unscaled.put_string_left(&format!("A{}", self.reroll_price),  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;

            y += FNTH;
            self.screen_geometry_unscaled.put_rect(v4(-1., y, FNTW/aspect* 8.0, FNTH*1.0), v4(0., 0., 1., 1.), 0.8, v4(0.5, 0.5, 0.5, 0.8), 0);
            self.screen_geometry_unscaled.put_string_left("cont.[t]",  -1.0, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); y += FNTH;
            
            let mut y = FNTH*2.0;
            let x = -1.0 + FNTW/aspect*9.0;
            self.screen_geometry_unscaled.put_rect(v4(x, y, FNTW/aspect* 30.0, FNTH*7.0), v4(0., 0., 1., 1.), 0.8, v4(0.5, 0.5, 0.5, 0.8), 0);

            y += FNTH;
            self.screen_geometry_unscaled.put_string_left("[1] ", x, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); self.disp_upgrade(x + 4.0*FNTW/aspect, y, FNTW/aspect, FNTH, 0.8, self.roll_upgrade(self.upgrade_seed[0])); y += FNTH;
            y += FNTH;
            self.screen_geometry_unscaled.put_string_left("[2] ", x, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); self.disp_upgrade(x + 4.0*FNTW/aspect, y, FNTW/aspect, FNTH, 0.8, self.roll_upgrade(self.upgrade_seed[1])); y += FNTH;
            y += FNTH;
            self.screen_geometry_unscaled.put_string_left("[3] ", x, y, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.)); self.disp_upgrade(x + 4.0*FNTW/aspect, y, FNTW/aspect, FNTH, 0.8, self.roll_upgrade(self.upgrade_seed[2])); y += FNTH;
            
            // warp zone panel

            // spinning player
            self.draw_player(0., 0., 0.1, 3.0*self.t, false);

            if self.t % 2.0 < 1.0 {
                self.screen_geometry_unscaled.put_string_centered("-- warp zone --",  0.0, -1.0, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.));
            }

            // draw cool white lines
            for i in 0..20 {
                let t_lines = self.t * 2.0;
                let si = khash(self.seed + i * 1293124147);
                let phase_i = krand(si) * (2.0*PI);
                let cycle_i = ((t_lines - phase_i) / (2.0*PI)).floor() as usize;
                let sc = khash(si + 1231237 * cycle_i);
                let x = krand(sc) * 2.0 - 1.0;
                let phase = (t_lines - phase_i) % (2.0*PI);
                let y = (((phase / (2.0*PI)) * 2.0 - 1.0) - 0.1)*1.1;
                self.screen_geometry_unscaled.put_rect(v4(x, y, 0.01, 0.1), v4(0., 0., 1., 1.,), 0.3, v4(1., 1., 1., 1.), 0);
            }


        } else {
            // normal level drawing
            self.draw_level();
            self.draw_player(self.player_pos.x, self.player_pos.y, 0.1, 0.0, true);
            self.draw_enemies();
            self.draw_gems();
            self.draw_enemy_projectiles();
            self.draw_player_projectiles();

            // self.screen_geometry.put_rect(v4(-1., -1., 1., 1.), v4(0., 0., 1., 1.), 0.2, v4(1., 1., 1., 1.), 5);

            // draw gem count
            self.screen_geometry_unscaled.put_rect(v4(-1., -1., 0.5, FNTH), v4(0., 0., 1., 1.), 0.1, v4(0., 0., 0., 1.), 0);
            self.screen_geometry_unscaled.put_rect(v4(-1., -1., 0.5*(self.player_hp/self.player_hp_max), FNTH), v4(0., 0., 1., 1.), 0.1, v4(1., 0., 0., 1.), 0);
            self.screen_geometry_unscaled.put_string_left(&format!("A{}", self.gems),  -1.0, -1.0 + FNTH, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.));
            // self.screen_geometry.put_string_left(&format!("hp: {} gems: {}",(self.player_hp * 100.0).round(),  self.gems),  -1.0, -1.0, FNTW, FNTH, 0.1, v4(1., 1., 1., 1.));
            
            if self.gems >= self.warp_price {
                    self.screen_geometry_unscaled.put_string_centered(&format!("warp available: A{} [t]", self.warp_price),  0.0, 1.0 - FNTH, FNTW/aspect, FNTH, 0.1, v4(1., 1., 1., 1.));
            }

        }
        // self.prod.push(Sound { id: 1, birthtime: self.t, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 440.0 * 3./2., freq_exp: 1.0, wait: 0.0, phase: 0.0 }).unwrap();


        self.gl.uniform_1_f32(self.gl.get_uniform_location(self.program, "time").as_ref(), self.t);

        self.gl.clear_color(0.5, 0.5, 0.5, 1.0);
        self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT); 
        self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
        self.gl.use_program(Some(self.program));
        self.gl.bind_vertex_array(Some(self.vao));
        self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));

        self.gl.uniform_matrix_4_f32_slice(self.gl.get_uniform_location(self.program, "projection").as_ref(), true, &[
            x_scale, 0., 0., -cam_x*x_scale,
            0., -1.*y_scale, 0., cam_y*y_scale,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ]);
        self.gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &self.world_geometry.buf, glow::DYNAMIC_DRAW);
        let vert_count = self.world_geometry.buf.len() / (10*4);
        self.gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);

        self.gl.uniform_matrix_4_f32_slice(self.gl.get_uniform_location(self.program, "projection").as_ref(), true, &[
            1./aspect, 0., 0., 1./aspect - 1.0,
            0., -1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ]);
        self.gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &self.screen_geometry.buf, glow::DYNAMIC_DRAW);
        let vert_count = self.screen_geometry.buf.len() / (10*4);
        self.gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);

        self.gl.uniform_matrix_4_f32_slice(self.gl.get_uniform_location(self.program, "projection").as_ref(), true, &[
            1., 0., 0., 0.,
            0., -1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ]);
        self.gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &self.screen_geometry_unscaled.buf, glow::DYNAMIC_DRAW);
        let vert_count = self.screen_geometry_unscaled.buf.len() / (10*4);
        self.gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);
        
        self.window.swap_buffers().unwrap();
        self.lmb_this_frame = false;
    }

    pub fn simulate(&mut self, dt: f32) {
        self.spawn_enemies_counter += dt;
        self.t_level += dt;

        self.update_player(dt);
        self.update_player_projectiles(dt);
        self.update_gems(dt);

        if self.spawn_enemies_counter > 1.0 {
            self.spawn_enemies_counter -= 1.0;
            self.spawn_enemies();
        }

        self.update_enemies(dt);
        self.update_enemy_projectiles(dt);
    }

    pub fn draw_level(&mut self) {
        let world_colour = v4(50.0, 0.3, 0.8, 1.0).hsv_to_rgb();
        // we could do some texturing on gpu
        // or just place rocks and shit as well as da gems
        self.world_geometry.put_rect(v4(-LEVEL_W/2.0, -LEVEL_H/2.0, LEVEL_W, LEVEL_H,), v4(0., 0., 1., 1.), 0.9, world_colour, 0);
    }    

    pub fn restart(&mut self) {
        self.player_pos = v2(0., 0.);
        self.player_vel = v2(0., 0.);
        self.player_succ = 0.5;
        self.player_hp = 1.0;
        self.player_hp_max = 1.0;
        self.player_hp_regen = 0.01;
        self.player_speed = 0.5;
        self.player_t_shoot = -999.0;
        self.player_cooldown = 0.7;
        self.player_damage = 0.5;
        self.player_proj_speed = 1.5;
        self.gems = 0;
        self.warp_price = 500;
        self.reroll_price = 10;
        self.seed = khash(self.seed + 1231237197);
        self.enemies_spawn_seed = khash(self.enemies_spawn_seed * 1231247 + 12312317);
        self.level_seed = khash(self.level_seed * 129371237 + 231417);
        self.zero_state();
        self.populate_with_gems();
    }

    pub fn zero_state(&mut self) {
        self.enemy_x.clear();
        self.enemy_y.clear();
        self.enemy_type.clear();
        self.enemy_birth_t.clear();
        self.enemy_attack_t.clear();
        self.enemy_hp.clear();
        self.enemy_projectile_x.clear();
        self.enemy_projectile_y.clear();
        self.enemy_projectile_vx.clear();
        self.enemy_projectile_vy .clear();
        self.enemy_projectile_type.clear();
        self.player_projectile_x.clear();
        self.player_projectile_y.clear();
        self.player_projectile_vx.clear();
        self.player_projectile_vy.clear();
        self.gem_x.clear();
        self.gem_y.clear();
        self.gem_vx.clear();
        self.gem_vy.clear();
        self.gem_type.clear();
    }
}




// ====================
// Audio stuff
// ====================
// 0 : kick drum
// 1 : sad ding

fn sample_next(o: &mut SampleRequestOptions) -> f32 {
    let mut acc = 0.0;
    let mut idx = o.sounds.len();
    loop {
        if idx == 0 {
            break;
        }
        idx -= 1;

        if o.sounds[idx].wait > 0.0 {
            o.sounds[idx].wait -= 1.0/44100.0;
            continue;
        }

        o.sounds[idx].elapsed += 1.0/44100.0;
        o.sounds[idx].remaining -= 1.0/44100.0;

        let t = o.sounds[idx].elapsed;

        if o.sounds[idx].remaining < 0.0 {
            o.sounds.swap_remove(idx);
            continue;
        }
        if o.sounds[idx].id == 0 {
            o.sounds[idx].magnitude *= 0.999;

            let f = o.sounds[idx].frequency;
            let f_trans = f*3.0;

            let t_trans = 1.0/(2.0*PI*f_trans);

            if o.sounds[idx].elapsed < t_trans {
                o.sounds[idx].phase += f_trans*2.0*PI*1.0/o.sample_rate;
            } else {
                o.sounds[idx].phase += f*2.0*PI*1.0/o.sample_rate;
            }
            // o.sounds[idx].phase += f*2.0*PI*1.0/o.sample_rate;

            //o.sounds[idx].phase = o.sounds[idx].phase % 2.0*PI; // this sounds really good lol

            acc += (o.sounds[idx].phase).sin() * o.sounds[idx].magnitude
        } else if o.sounds[idx].id == 1 {
            o.sounds[idx].magnitude *= o.sounds[idx].mag_exp;
            o.sounds[idx].frequency *= o.sounds[idx].freq_exp;
            o.sounds[idx].phase += o.sounds[idx].frequency*2.0*PI*1.0/o.sample_rate;
            acc += (o.sounds[idx].phase).sin() * o.sounds[idx].magnitude
        } else if o.sounds[idx].id == 2 {
            o.sounds[idx].magnitude *= o.sounds[idx].mag_exp;
            acc += krand(o.sounds[idx].samp as usize) * o.sounds[idx].magnitude;
        }
        o.sounds[idx].samp += 1;
    }
    acc
}



pub struct SampleRequestOptions {
    pub sample_rate: f32,
    pub nchannels: usize,
    pub channel: Consumer<Sound>,
    pub sounds: Vec<Sound>,
}

pub fn stream_setup_for<F>(on_sample: F, channel: Consumer<Sound>) -> Result<cpal::Stream, anyhow::Error>
where
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let (_host, device, config) = host_device_setup()?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => stream_make::<f32, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::I16 => stream_make::<i16, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::U16 => stream_make::<u16, _>(&device, &config.into(), on_sample, channel),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {:?}", config);

    Ok((host, device, config))
}


pub fn stream_make<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    on_sample: F,
    channel: Consumer<Sound>,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let sample_rate = config.sample_rate.0 as f32;
    let nchannels = config.channels as usize;
    let mut request = SampleRequestOptions {
        sample_rate,
        nchannels,
        sounds: vec![],
        channel,
    };
    let err_fn = |err| eprintln!("Error building output sound stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            on_window(output, &mut request, on_sample)
        },
        err_fn,
    )?;

    Ok(stream)
}

fn on_window<T, F>(output: &mut [T], request: &mut SampleRequestOptions, mut on_sample: F)
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static,
{
    if let Some(sc) = request.channel.pop() {
        request.sounds.push(sc);
    }
    for frame in output.chunks_mut(request.nchannels) {
        let value: T = cpal::Sample::from::<f32>(&on_sample(request));
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}



// move this audio shit away
// warp out
// warp in
// shop
// shop sfx
// shop music etc