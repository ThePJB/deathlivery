#version 330 core
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec4 in_colour;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in uint in_fs_mode;

uniform mat4 projection;

// const mat4 projection = mat4(
//     1, 0, 0, 0,
//     0, -1, 0, 0,
//     0, 0, 1, 0,
//     0, 0, 0, 1
// );

out vec4 colour;
out vec2 uv;
flat out uint fs_mode;

void main() {
    colour = in_colour;
    uv = in_uv;
    fs_mode = in_fs_mode;
    gl_Position = projection * vec4(in_pos, 1.0);
}