/*
 * Copyright © 2024 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#version 450

#define COLOR 0.0, 0.8, 0.2
#define POSITION vec2(3.1, -2.0)
#define ANGLE (-2.0 * angle - 9.0)

layout(set = 0, binding = 0) uniform block {
    uniform mat4 projection;
};

layout(push_constant) uniform constants
{
    uniform float angle, view_rot_0, view_rot_1, h;
};

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_color;

const vec3 L = normalize(vec3(5.0, 5.0, 10.0));
const vec3 material_color = vec3(COLOR);
const float PI = radians(180);

mat4
mat4_rotate(mat4 m, float angle, float x, float y, float z)
{
   float s = sin(angle);
   float c = cos(angle);
   mat4 r = mat4(
      x * x * (1 - c) + c,     y * x * (1 - c) + z * s, x * z * (1 - c) - y * s, 0,
      x * y * (1 - c) - z * s, y * y * (1 - c) + c,     y * z * (1 - c) + x * s, 0,
      x * z * (1 - c) + y * s, y * z * (1 - c) - x * s, z * z * (1 - c) + c,     0,
      0, 0, 0, 1
   );

   return m * r;
}

mat4
mat4_translate(mat4 m, float x, float y, float z)
{
   mat4 t = mat4( 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  x, y, z, 1 );

   return m * t;
}

mat4
mat4_frustum_vk(mat4 m, float l, float r, float b, float t, float n, float f)
{
   mat4 tmp = mat4(1.0);

   float deltaX = r - l;
   float deltaY = t - b;
   float deltaZ = f - n;

   tmp[0].x = (2 * n) / deltaX;
   tmp[1].y = (-2 * n) / deltaY;
   tmp[2].x = (r + l) / deltaX;
   tmp[2].y = (t + b) / deltaY;
   tmp[2].z = f / (n - f);
   tmp[2].w = -1;
   tmp[3].z = -(f * n) / deltaZ;
   tmp[3].w = 0;

   return tmp;
}


void main()
{
   mat4 view = mat4(1.0);
   view = mat4_translate(view, 0, 0, -40);
   view = mat4_rotate(view, 2 * PI * view_rot_0 / 360.0, 1, 0, 0);
   view = mat4_rotate(view, 2 * PI * view_rot_1 / 360.0, 0, 1, 0);
   view = mat4_rotate(view, 0, 0, 0, 1);

   /* Translate and rotate the gear */
   mat4 modelview = mat4(1.0) * view;
   modelview = mat4_translate(modelview, POSITION.x, POSITION.y, 0);
   modelview = mat4_rotate(modelview, 2 * PI * ANGLE / 360.0, 0, 0, 1);

   mat4 projection = mat4(1.0);
   projection = mat4_frustum_vk(projection, -1.0, 1.0, -h, +h, 5.0f, 60.0f);

    vec3 N = normalize(mat3(modelview) * in_normal);

    float diffuse = max(0.0, dot(N, L));
    float ambient = 0.2;
    out_color = vec4((ambient + diffuse) * material_color, 1.0);

    gl_Position = projection * (modelview * in_position);
}
