mod math;
mod game;
mod vertex_buffer;
mod enemies;
mod player;
mod sound;
mod terrain;
mod kimg;

use std::env;


fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    unsafe {
        let event_loop = glutin::event_loop::EventLoop::new();
        let mut game = game::Game::new(&event_loop);
        event_loop.run(move |event, _, _| game.handle_event(event));
    }
    loop {}
}