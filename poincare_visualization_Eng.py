import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def linear_retarder_matrix_general(theta, delta):
    """
    Compute the general Mueller matrix of an ideal linear retarder.
    θ: fast axis angle relative to horizontal (radians)
    δ: phase retardance (radians)
    """
    c2t = np.cos(2 * theta)
    s2t = np.sin(2 * theta)
    cosd = np.cos(delta)
    sind = np.sin(delta)

    M = np.array([
        [1,       0,                     0,                  0],
        [0,  c2t**2 + cosd * s2t**2, (1 - cosd) * c2t * s2t,    -sind * s2t],
        [0, (1 - cosd) * c2t * s2t, cosd * c2t**2 + s2t**2,     sind * c2t],
        [0,       sind * s2t,           -sind * c2t,           cosd]
    ])
    return M


def plot_stokes_on_poincare_interactive(stokes_vectors, label_list=None):
    stokes_vectors = np.atleast_2d(stokes_vectors)
    if stokes_vectors.shape[1] != 4:
        raise ValueError("Each Stokes vector must contain 4 components: [S0, S1, S2, S3]")

    norm_vectors = stokes_vectors[:, 1:] / stokes_vectors[:, 0:1]

    # Create Poincaré sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.2,
        colorscale='Blues',
        showscale=False
    )

    special_points = {
        "H (Horizontal)": [1, 0, 0],
        "V (Vertical)": [-1, 0, 0],
        "+45° Linear": [0, 1, 0],
        "-45° Linear": [0, -1, 0],
        "R (Right Circular)": [0, 0, 1],
        "L (Left Circular)": [0, 0, -1],
    }

    reference_markers = []
    for label, (sx, sy, sz) in special_points.items():
        reference_markers.append(go.Scatter3d(
            x=[sx],
            y=[sy],
            z=[sz],
            mode='markers+text',
            marker=dict(size=5, color='gray', opacity=0.6),
            text=[label],
            textposition="top center",
            name=label,
            showlegend=False
        ))

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

    vector_traces = []
    for i, vec in enumerate(norm_vectors):
        color = colors[i % len(colors)]
        label = label_list[i] if label_list and i < len(label_list) else f"Vec {i+1}"

        trace = go.Scatter3d(
            x=[0, vec[0]],
            y=[0, vec[1]],
            z=[0, vec[2]],
            mode='lines+markers+text',
            marker=dict(size=5, color=color),
            line=dict(width=4, color=color),
            text=[None, label],
            textposition="top center",
            name=label,
            showlegend=True
        )
        vector_traces.append(trace)

    fig = go.Figure(data=[surface] + reference_markers + vector_traces)

    fig.update_layout(
        title="Stokes Vectors on the Poincaré Sphere (Multiple Transformations)",
        scene=dict(
            xaxis_title="S1",
            yaxis_title="S2",
            zaxis_title="S3",
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(title="Vector Labels")
    )

    fig.show()


def plot_stokes_with_transformed_vector(original_vectors, transformed_vector):
    all_vectors = np.vstack([original_vectors, transformed_vector])
    plot_stokes_on_poincare_interactive(all_vectors)


def user_input_single_transformation_multi():
    S_in = np.array(eval(input("Enter the Stokes vector (e.g., [1, 1, 0, 0]): ")))
    theta_deg_list = eval(input("Enter one or more θ values (degrees), e.g., [0, 45, 90]: "))
    delta_deg_list = eval(input("Enter one or more δ values (degrees), e.g., [90, 180, 270]: "))

    if len(theta_deg_list) != len(delta_deg_list):
        print("The number of θ and δ values must be equal.")
        return

    all_vectors = [S_in]
    labels = ["Input Vector"]

    print("Transformation Results:")
    for i, (theta_deg, delta_deg) in enumerate(zip(theta_deg_list, delta_deg_list)):
        theta_rad = np.deg2rad(theta_deg)
        delta_rad = np.deg2rad(delta_deg)

        M = linear_retarder_matrix_general(theta_rad, delta_rad)
        S_out = M @ S_in
        all_vectors.append(S_out)

        label = f"θ={theta_deg}°, δ={delta_deg}°"
        labels.append(label)

        print(f"[{label}] -> Output Vector: {np.round(S_out, 3)}")

    all_vectors = np.vstack(all_vectors)
    plot_stokes_on_poincare_interactive(all_vectors, labels)
def user_input_sweep_transformation():
    S_in = np.array(eval(input("Enter the Stokes vector (e.g., [1, 1, 0, 0]): ")))
    mode = input("Select sweep mode: fix θ or fix δ? Type 'theta' or 'delta': ").strip().lower()

    all_paths = []
    cmap = cm.get_cmap('hsv', 20)  # Up to 20 unique colors
    color_list = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b, _ in cmap(np.linspace(0, 1, 20))]

    if mode == "theta":
        theta_list = eval(input("Enter one or more θ values in degrees (e.g., [0, 45, 90]): "))
        delta_deg_list = np.arange(0, 216, 5)

        for i, theta_deg in enumerate(theta_list):
            theta_rad = np.deg2rad(theta_deg)
            transformed = []

            for delta_deg in delta_deg_list:
                delta_rad = np.deg2rad(delta_deg)
                M = linear_retarder_matrix_general(theta_rad, delta_rad)
                S_out = M @ S_in
                transformed.append(S_out)

            transformed = np.array(transformed)
            norm_coords = transformed[:, 1:] / transformed[:, 0:1]

            path = go.Scatter3d(
                x=norm_coords[:, 0],
                y=norm_coords[:, 1],
                z=norm_coords[:, 2],
                mode='lines',
                line=dict(width=4, color=color_list[i % len(color_list)]),
                name=f"θ = {theta_deg}°"
            )
            all_paths.append(path)

    elif mode == "delta":
        delta_list = eval(input("Enter one or more δ values in degrees (e.g., [90, 180]): "))
        theta_deg_list = np.arange(0, 180, 5)

        for i, delta_deg in enumerate(delta_list):
            delta_rad = np.deg2rad(delta_deg)
            transformed = []

            for theta_deg in theta_deg_list:
                theta_rad = np.deg2rad(theta_deg)
                M = linear_retarder_matrix_general(theta_rad, delta_rad)
                S_out = M @ S_in
                transformed.append(S_out)

            transformed = np.array(transformed)
            norm_coords = transformed[:, 1:] / transformed[:, 0:1]

            path = go.Scatter3d(
                x=norm_coords[:, 0],
                y=norm_coords[:, 1],
                z=norm_coords[:, 2],
                mode='lines',
                line=dict(width=4, color=color_list[i % len(color_list)]),
                name=f"δ = {delta_deg}°"
            )
            all_paths.append(path)

    else:
        print("Invalid input. Please enter 'theta' or 'delta'.")
        return

    # Plot
    sphere_surface = generate_poincare_sphere()
    special_points = generate_special_points()
    input_vector = go.Scatter3d(
        x=[S_in[1] / S_in[0]],
        y=[S_in[2] / S_in[0]],
        z=[S_in[3] / S_in[0]],
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=["Input Vector"],
        textposition="top center",
        name="Input",
        showlegend=False
    )

    fig = go.Figure(data=[sphere_surface] + special_points + [input_vector] + all_paths)
    fig.update_layout(
        title="Retarder Transformation Trajectories (Multiple Parameter Sets)",
        scene=dict(
            xaxis_title="S1",
            yaxis_title="S2",
            zaxis_title="S3",
            aspectmode='data',
        ),
        legend=dict(title="Trajectories"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def generate_poincare_sphere():
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    surface = go.Surface(
        x=x, y=y, z=z,
        opacity=0.2,
        colorscale='Blues',
        showscale=False
    )
    return surface


def generate_special_points():
    special_points = {
        "H (Horizontal)": [1, 0, 0],
        "V (Vertical)": [-1, 0, 0],
        "+45° Linear": [0, 1, 0],
        "-45° Linear": [0, -1, 0],
        "R (Right Circular)": [0, 0, 1],
        "L (Left Circular)": [0, 0, -1],
    }
    markers = []
    for label, (sx, sy, sz) in special_points.items():
        markers.append(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode='markers+text',
            marker=dict(size=6, color='red'),
            text=[label],
            textposition='top center',
            name=label,
            showlegend=False
        ))
    return markers


if __name__ == "__main__":
    print("Select a function:")
    print("1 - Single transformation: show input and multiple outputs")
    print("2 - Sweep θ or δ and plot trajectories")
    choice = input("Enter function number (1 or 2): ")

    if choice == "1":
        user_input_single_transformation_multi()
    elif choice == "2":
        user_input_sweep_transformation()
    else:
        print("Invalid selection.")
