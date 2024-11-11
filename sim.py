import taichi as ti
import time
import math
import platform

if platform.system() == 'Darwin':
    ti.init(arch=ti.metal)
elif ti.has_gpu():
    ti.init(arch=ti.vulkan)
else:
    ti.init(arch=ti.cpu)

screen_width = 1920
screen_height = 1080
aspect_ratio = screen_width / screen_height
diagonal = math.sqrt(screen_width ** 2 + screen_height ** 2)
sim_size = min(screen_width, screen_height)
num_part = 200_000
max_particles = 200_000
num_particles = ti.field(dtype=ti.i32, shape=())
num_particles[None] = num_part
radii = 4
masses = 1
if radii == 1:
    cell_size = 3.2 * radii
else:
    cell_size = 2.0 * radii
num_substeps = 32
dt = 1 / 60 / num_substeps

restitution = 0.8
response_coef = 0.75
repulsion_coef = 0
damping_factor = 1
gravity_y = 1000.0

# using ti.fields is faster than passing arguments to the functions
gravity = ti.field(dtype=ti.f32, shape=())

positions = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
positions_old = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
accelerations = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
min_velocity = ti.field(dtype=ti.f32, shape=max_particles)
max_velocity = ti.field(dtype=ti.f32, shape=max_particles)
colors = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)

constraint_center = ti.Vector([sim_size // 2, sim_size // 2, sim_size // 2], dt=ti.f32)

att_point = ti.Vector.field(3, dtype=ti.f32, shape=())
att_force = ti.field(dtype=ti.f32, shape=())
att_enabled = ti.field(dtype=ti.i32, shape=())
cont_shape = ti.field(dtype=ti.i32, shape=())
cont_size = ti.field(dtype=ti.i32, shape=())
add_particles = ti.field(dtype=ti.i32, shape=())
spawn_acc = ti.field(dtype=ti.f32, shape=())

grid_size = int((sim_size + cell_size - 1) // cell_size)
grid = ti.field(dtype=ti.i32, shape=(grid_size, grid_size, grid_size, 8))
grid_counts = ti.field(dtype=ti.i32, shape=(grid_size, grid_size, grid_size))

num_vertices = 18
vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(dtype=ti.i32, shape=num_vertices)

window = ti.ui.Window("Particular", (screen_width, screen_height), pos=(0, 30))
canvas = window.get_canvas()
gui = window.get_gui()
scene = window.get_scene()
camera = ti.ui.Camera()


@ti.kernel
def initialize():
    for i in range(num_particles[None]):
        positions[i] = [ti.random() * sim_size, ti.random() * sim_size, ti.random() * sim_size]
        positions_old[i] = positions[i]
        velocities[i] = [0.0, 0.0, 0.0]
        accelerations[i] = [0.0, 0.0, 0.0]
        colors[i] = [0.25, 0.25, 1.0]
        min_velocity[i] = 1e-10
        max_velocity[i] = 1e10
    for i in range(max_particles):
        normalized_positions[i] = [-1.0, -1.0, -1.0]
    gravity[None] = gravity_y
    spawn_acc[None] = 1200.0
    att_enabled[None] = False
    att_force[None] = 1200.0
    cont_size[None] = (sim_size // 2)
    cont_shape[None] = 1

    # camera.position(1.25, 1.06, 1.55)
    # camera.lookat(0.71, 0.67, 0.80)
    # camera.fov(55.0)
    camera.position(2.15, 1.8, 2.15)
    camera.lookat(1.55, 1.285, 1.55)
    camera.fov(35.0)
    plane_positive_extent = 10.0  # Extent in the positive x and z directions
    plane_negative_extent = 0.2  # Extent in the negative x and z directions
    plane_height = 1.0  # Extent in the y direction
    vertices[0] = ti.Vector([-plane_negative_extent, -plane_height, 0.0])
    vertices[1] = ti.Vector([plane_positive_extent, -plane_height, 0.0])
    vertices[2] = ti.Vector([plane_positive_extent, plane_height, 0.0])
    vertices[3] = ti.Vector([-plane_negative_extent, plane_height, 0.0])

    indices[0] = 0
    indices[1] = 1
    indices[2] = 2
    indices[3] = 2
    indices[4] = 3
    indices[5] = 0

    # Vertices and indices for the z-plane
    vertices[4] = ti.Vector([0.0, -plane_height, -plane_negative_extent])
    vertices[5] = ti.Vector([0.0, -plane_height, plane_positive_extent])
    vertices[6] = ti.Vector([0.0, plane_height, plane_positive_extent])
    vertices[7] = ti.Vector([0.0, plane_height, -plane_negative_extent])

    indices[6] = 4
    indices[7] = 5
    indices[8] = 6
    indices[9] = 6
    indices[10] = 7
    indices[11] = 4

    # Vertices and indices for the y-plane
    vertices[8] = ti.Vector([-plane_negative_extent, 0.0, -plane_negative_extent])
    vertices[9] = ti.Vector([plane_positive_extent, 0.0, -plane_negative_extent])
    vertices[10] = ti.Vector([plane_positive_extent, 0.0, plane_positive_extent])
    vertices[11] = ti.Vector([-plane_negative_extent, 0.0, plane_positive_extent])

    indices[12] = 8
    indices[13] = 9
    indices[14] = 10
    indices[15] = 10
    indices[16] = 11
    indices[17] = 8


@ti.kernel
def set_attraction_point(x: ti.f32, y: ti.f32):
    z = 1.0 - x
    att_point[None] = [x * sim_size, (1.0 - y) * sim_size, z * sim_size]


@ti.kernel
def normalize_positions():
    for i in range(num_particles[None]):
        normalized_positions[i] = positions[i].x / sim_size, 1.0 - positions[i].y / sim_size, positions[i].z / sim_size


@ti.kernel
def apply_gravity():
    for i in range(num_particles[None]):
        accelerations[i].y += gravity[None]
        if att_enabled[None]:
            direction = att_point[None] - positions[i]
            distance = direction.norm() + 1e-10
            normalized_direction = direction / distance
            accelerations[i] += att_force[None] * normalized_direction


@ti.kernel
def update_positions():
    for i in range(num_particles[None]):
        temp_position = positions[i]
        displacements = positions[i] - positions_old[i]
        positions[i] += displacements + accelerations[i] * dt * dt
        positions_old[i] = temp_position
        velocities[i] = (positions[i] - positions_old[i]) / dt
        accelerations[i] = [0.0, 0.0, 0.0]


@ti.kernel
def apply_constraint():
    for i in range(num_particles[None]):
        if cont_shape[None] == 0:
            to_center = constraint_center - positions[i]
            distance = to_center.norm()
            if distance > (cont_size[None] - radii):
                direction = to_center / distance
                correction = direction * (distance - (cont_size[None] - radii))
                positions[i] += (1 + restitution) * correction
        else:
            for j in ti.static(range(3)):
                if positions[i][j] < constraint_center[j] - cont_size[None] + radii:
                    overstep = constraint_center[j] - cont_size[None] + radii - positions[i][j]
                    positions[i][j] += (1 + restitution) * overstep

                elif positions[i][j] > constraint_center[j] + cont_size[None] - radii:
                    overstep = positions[i][j] - (constraint_center[j] + cont_size[None] - radii)
                    positions[i][j] -= (1 + restitution) * overstep


@ti.kernel
def update_grid():
    # Clear the grid
    for i, j, k in grid_counts:
        grid_counts[i, j, k] = 0

    # Populate the grid with particles
    for p in range(num_particles[None]):
        cell_x = ti.cast(positions[p].x // cell_size, ti.i32)
        cell_y = ti.cast(positions[p].y // cell_size, ti.i32)
        cell_z = ti.cast(positions[p].z // cell_size, ti.i32)

        cell_x = ti.max(0, ti.min(cell_x, grid_size - 1))
        cell_y = ti.max(0, ti.min(cell_y, grid_size - 1))
        cell_z = ti.max(0, ti.min(cell_z, grid_size - 1))

        if grid_counts[cell_x, cell_y, cell_z] < 8:  # Prevent overflow
            count = ti.atomic_add(grid_counts[cell_x, cell_y, cell_z], 1)
            grid[cell_x, cell_y, cell_z, count] = p


@ti.kernel
def handle_collisions():
    for i, j, k in ti.ndrange(grid_size, grid_size, grid_size):
        count = grid_counts[i, j, k]
        for m in range(count):
            p1 = grid[i, j, k, m]
            for ii, jj, kk in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                ni = i + ii
                nj = j + jj
                nk = k + kk
                if 0 <= ni < grid_size and 0 <= nj < grid_size and 0 <= nk < grid_size:
                    neighbor_count = grid_counts[ni, nj, nk]
                    for n in range(neighbor_count):
                        p2 = grid[ni, nj, nk, n]
                        if p1 < p2:
                            delta = positions[p1] - positions[p2]
                            dist_sq = delta.dot(delta)
                            min_dist = radii + radii
                            if dist_sq < min_dist * min_dist:
                                dist = ti.sqrt(dist_sq)
                                normalized_delta = delta / dist
                                relative_velocity = velocities[p1] - velocities[p2]
                                vn = relative_velocity.dot(normalized_delta)
                                if vn < 0:
                                    mass_ratio_1 = radii / (radii + radii)
                                    mass_ratio_2 = radii / (radii + radii)
                                    delta = 0.5 * response_coef * (dist - min_dist)
                                    positions[p1] -= normalized_delta * (mass_ratio_2 * delta)
                                    positions[p2] += normalized_delta * (mass_ratio_1 * delta)
                                    repulsion = repulsion_coef * normalized_delta
                                    positions[p1] += repulsion
                                    positions[p2] -= repulsion


@ti.kernel
def compute_colors():
    blue = ti.Vector([0.0, 0.27, 0.71])
    green = ti.Vector([0.0, 0.78, 0.47])
    yellow = ti.Vector([1.0, 0.92, 0.0])
    red = ti.Vector([1.0, 0.26, 0.0])

    scaling_factor = 0.56e7

    for i in range(num_particles[None]):
        min_velocity[i] = min(min_velocity[i], velocities[i].norm())
        max_velocity[i] = max(max_velocity[i], velocities[i].norm())

        normalized_velocity = (velocities[i].norm() - min_velocity[i]) / (
                max_velocity[i] - min_velocity[i])
        scaled_velocity = min(normalized_velocity * scaling_factor, 1.0)

        if scaled_velocity < 0.33:
            t = scaled_velocity / 0.33
            colors[i] = (1 - t) * blue + t * green
        elif scaled_velocity < 0.66:
            t = (scaled_velocity - 0.33) / 0.33
            colors[i] = (1 - t) * green + t * yellow
        else:
            t = (scaled_velocity - 0.66) / 0.34
            colors[i] = (1 - t) * yellow + t * red


@ti.kernel
def add_particles(n: ti.i32):
    start = num_particles[None]
    end = min(num_particles[None] + n, max_particles)
    for i in range(start, end):
        positions[i] = [screen_width / 2, screen_height / 2 - 200, (screen_width + screen_height) / 4]

        direction = att_point[None]
        # direction = ti.Vector([-12.0, -49.0, 0.0])

        # Introduce some randomness to the direction
        random_vec = ti.Vector([ti.random(), ti.random(), ti.random()]) * 0.2
        direction += random_vec
        direction_normalized = direction.normalized()

        velocities[i] = direction_normalized * spawn_acc[None]

        positions_old[i] = positions[i] - (velocities[i] * dt)

        accelerations[i] = [0.0, 0.0, 0.0]
        colors[i] = [0.25, 0.25, 1.0]
        min_velocity[i] = 1e-10
        max_velocity[i] = 1e10

    num_particles[None] = end


@ti.kernel
def remove_particles():
    if num_particles[None] > 0:
        last_index = num_particles[None] - 1

        positions[last_index] = [-10.0, -10.0, -10.0]
        positions_old[last_index] = positions[last_index]
        normalized_positions[last_index] = [-10.0, -10.0, -10.0]
        velocities[last_index] = [0.0, 0.0, 0.0]
        accelerations[last_index] = [0.0, 0.0, 0.0]
        colors[last_index] = [0.0, 0.0, 0.0]
        min_velocity[last_index] = 1e-10
        max_velocity[last_index] = 1e10

        num_particles[None] -= 1


particle_add_interval = 0.01
last_add_time = time.time()

initialize()

while window.running:
    current_time = time.time()
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == ti.ui.RMB:
            att_enabled[None] = True

    for e in window.get_events(ti.ui.RELEASE):
        if e.key == ti.ui.RMB:
            att_enabled[None] = False
    if window.is_pressed(ti.ui.SPACE):
        if current_time - last_add_time >= particle_add_interval:
            num_new_particles = 1
            add_particles(num_new_particles)
            last_add_time = current_time
    elif window.is_pressed(ti.ui.SHIFT):
        if current_time - last_add_time >= particle_add_interval:
            remove_particles()
            last_add_time = current_time
    elif window.is_pressed(ti.ui.RMB):
        set_attraction_point(*window.get_cursor_pos())

    with gui.sub_window("Simulation Options", 0.01, 0.01, 0.26, 0.18):
        np = str(num_particles[None])
        gui.text(f"Particle Count: {np}")
        gravity[None] = gui.slider_float("Gravity", gravity[None], -2000, 2000)
        att_force[None] = gui.slider_float("Pull force", att_force[None], 0, 15000)
        spawn_acc[None] = gui.slider_float("Spawn acc", spawn_acc[None], 0, 5000)
        if gui.button("Zero Gravity"):
            gravity[None] = 0.0
        if gui.button("Default Gravity"):
            gravity[None] = 1000.0
    with gui.sub_window("Container Options", 0.75, 0.01, 0.24, 0.13):
        cont_size[None] = gui.slider_int("size", cont_size[None], 50, sim_size // 2)
        if gui.button("Default Size"):
            cont_size[None] = sim_size // 2
        if gui.button("Sphere"):
            cont_shape[None] = 0
        if gui.button("Box"):
            cont_shape[None] = 1

    for _ in range(num_substeps):
        apply_gravity()
        update_positions()
        apply_constraint()
        update_grid()
        handle_collisions()
        normalize_positions()
        compute_colors()

    scene.mesh(vertices, indices, two_sided=True, color=(0.06, 0.06, 0.06))
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(normalized_positions, radius=radii / diagonal, per_vertex_color=colors)
    scene.point_light(pos=(0.5, 1.0, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(1.0, 5.0, 2.5), color=(1, 1, 1))
    canvas.scene(scene)
    window.show()