import math


def sun_height_over_slope(slope_deg, aspect_deg, sun_elev_deg, sun_az_deg):
    beta = math.radians(slope_deg)
    gamma = math.radians(aspect_deg)
    alpha = math.radians(sun_elev_deg)
    A = math.radians(sun_az_deg)

    cos_theta = math.sin(alpha) * math.cos(beta) + math.cos(alpha) * math.sin(
        beta
    ) * math.cos(A - gamma)
    # numerisch clampen
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))  # Inzidenzwinkel
    height_over_plane = 90.0 - theta
    return height_over_plane, theta


# Beispiel:
h, theta = sun_height_over_slope(67.9, 199.8, 19.06, 193.7)
print("Sonnenhöhe über Hang:", h, "deg; Inzidenzwinkel:", theta, "deg")
