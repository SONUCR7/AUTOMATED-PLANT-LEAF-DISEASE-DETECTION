"""
Sample Leaf Image Generator for Demonstration & Testing
=======================================================
Synthesizes visually authentic leaf images with realistic morphology,
vein structures, chlorotic halos, and necrotic lesion patterns (bullseyes,
water-soaked patches, bacterial specks) on neutral backgrounds, matching
PlantVillage dataset characteristics.
"""

import os
import cv2
import numpy as np


def create_base_canvas(size=(512, 512), bg_color=(45, 45, 48)):
    """Creates a neutral studio background canvas similar to PlantVillage."""
    canvas = np.full((size[0], size[1], 3), bg_color, dtype=np.uint8)
    # Add subtle background texture noise
    noise = np.random.normal(0, 3, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def draw_potato_leaf_shape(canvas, center=(256, 260), scale=(130, 180), angle=-10, green_color=(40, 145, 45)):
    """Draws an ovate potato leaflet with main and lateral veins."""
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, scale, angle, 0, 360, 255, -1)
    
    # Base green leaf fill with realistic gradient
    y, x = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    leaf_pixels = mask > 0
    
    # Slight color variations
    canvas[leaf_pixels] = green_color
    texture_noise = np.random.normal(0, 7, (canvas.shape[0], canvas.shape[1], 3)).astype(np.int16)
    canvas[leaf_pixels] = np.clip(canvas[leaf_pixels].astype(np.int16) + texture_noise[leaf_pixels], 0, 255).astype(np.uint8)
    
    # Draw central midrib and lateral veins
    pt1 = (center[0] - int(np.sin(np.radians(angle)) * scale[1] * 0.9),
           center[1] + int(np.cos(np.radians(angle)) * scale[1] * 0.9))
    pt2 = (center[0] + int(np.sin(np.radians(angle)) * scale[1] * 0.9),
           center[1] - int(np.cos(np.radians(angle)) * scale[1] * 0.9))
    cv2.line(canvas, pt1, pt2, (60, 175, 60), 3)
    
    # Lateral veins
    for t in np.linspace(0.2, 0.8, 6):
        vx = int(pt1[0] + t * (pt2[0] - pt1[0]))
        vy = int(pt1[1] + t * (pt2[1] - pt1[1]))
        cv2.line(canvas, (vx, vy), (vx - 60, vy - 20), (55, 165, 55), 2)
        cv2.line(canvas, (vx, vy), (vx + 60, vy - 15), (55, 165, 55), 2)
        
    return mask


def draw_tomato_leaf_shape(canvas, center=(256, 260), scale=(120, 190), angle=5, green_color=(35, 135, 40)):
    """Draws a lobed/serrated tomato leaflet with prominent veins."""
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, scale, angle, 0, 360, 255, -1)
    # Add lobes
    cv2.ellipse(mask, (center[0] - 80, center[1] + 30), (50, 40), 30, 0, 360, 255, -1)
    cv2.ellipse(mask, (center[0] + 80, center[1] + 25), (50, 40), -30, 0, 360, 255, -1)
    cv2.ellipse(mask, (center[0] - 65, center[1] - 50), (45, 35), 40, 0, 360, 255, -1)
    cv2.ellipse(mask, (center[0] + 65, center[1] - 55), (45, 35), -40, 0, 360, 255, -1)
    
    leaf_pixels = mask > 0
    canvas[leaf_pixels] = green_color
    noise = np.random.normal(0, 8, canvas.shape).astype(np.int16)
    canvas[leaf_pixels] = np.clip(canvas[leaf_pixels].astype(np.int16) + noise[leaf_pixels], 0, 255).astype(np.uint8)
    
    # Midrib
    cv2.line(canvas, (center[0], center[1] + scale[1] - 20), (center[0], center[1] - scale[1] + 20), (55, 165, 55), 3)
    return mask


def draw_pepper_leaf_shape(canvas, center=(256, 260), scale=(110, 190), angle=0, green_color=(30, 125, 35)):
    """Draws an acuminated bell pepper leaf with smooth margins."""
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, scale, angle, 0, 360, 255, -1)
    # Acuminate pointed tip
    tip_pts = np.array([
        [center[0] - 35, center[1] - scale[1] + 30],
        [center[0] + 35, center[1] - scale[1] + 30],
        [center[0], center[1] - scale[1] - 30]
    ], np.int32)
    cv2.fillPoly(mask, [tip_pts], 255)
    
    leaf_pixels = mask > 0
    canvas[leaf_pixels] = green_color
    noise = np.random.normal(0, 6, canvas.shape).astype(np.int16)
    canvas[leaf_pixels] = np.clip(canvas[leaf_pixels].astype(np.int16) + noise[leaf_pixels], 0, 255).astype(np.uint8)
    
    # Veins
    cv2.line(canvas, (center[0], center[1] + scale[1] - 20), (center[0], center[1] - scale[1]), (50, 150, 50), 3)
    return mask


def add_early_blight_bullseyes(canvas, leaf_mask, centers):
    """Adds characteristic Alternaria concentric rings (dark necrotic bullseye with chlorotic yellow halo)."""
    for cx, cy, rad in centers:
        if leaf_mask[cy, cx] == 0:
            continue
        # Outer yellow chlorotic halo
        halo_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.circle(halo_mask, (cx, cy), int(rad * 1.55), 255, -1)
        halo_pixels = (halo_mask > 0) & (leaf_mask > 0)
        canvas[halo_pixels] = (205, 195, 45) # Yellow-green chlorosis
        
        # Outer necrotic brown ring
        ring_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.circle(ring_mask, (cx, cy), rad, 255, -1)
        ring_pixels = (ring_mask > 0) & (leaf_mask > 0)
        canvas[ring_pixels] = (100, 60, 25) # Dark brown
        
        # Inner concentric rings
        cv2.circle(canvas, (cx, cy), int(rad * 0.75), (135, 85, 35), 2)
        cv2.circle(canvas, (cx, cy), int(rad * 0.5), (75, 45, 20), -1)
        cv2.circle(canvas, (cx, cy), int(rad * 0.25), (45, 25, 10), -1)


def add_late_blight_lesions(canvas, leaf_mask, spots):
    """Adds irregular, water-soaked, spreading dark brown/black lesions on leaf margins."""
    for cx, cy, rx, ry in spots:
        lesion_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.ellipse(lesion_mask, (cx, cy), (rx, ry), 25, 0, 360, 255, -1)
        # Dilate irregularly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        lesion_mask = cv2.dilate(lesion_mask, kernel)
        
        active = (lesion_mask > 0) & (leaf_mask > 0)
        # Water-soaked dark necrotic center
        canvas[active] = (50, 40, 30)
        
        # Pale chlorotic margin
        edge = cv2.Canny(lesion_mask, 50, 150)
        edge_active = (cv2.dilate(edge, np.ones((5, 5))) > 0) & (leaf_mask > 0)
        canvas[edge_active] = (175, 165, 55)


def add_bacterial_spots(canvas, leaf_mask, num_spots=45):
    """Adds numerous small circular necrotic specks with distinctive yellow halos (Xanthomonas)."""
    np.random.seed(101)
    y_idxs, x_idxs = np.where(leaf_mask > 0)
    rand_indices = np.random.choice(len(y_idxs), size=num_spots, replace=False)
    
    for idx in rand_indices:
        cx, cy = x_idxs[idx], y_idxs[idx]
        rad = np.random.randint(4, 9)
        # Yellow halo
        cv2.circle(canvas, (cx, cy), int(rad * 1.8), (210, 200, 45), -1)
        # Dark brown / black sunken center
        cv2.circle(canvas, (cx, cy), rad, (45, 30, 20), -1)


def generate_all_samples(output_dir="dataset/samples"):
    """Generates all sample benchmark leaf images for instant viva demo."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[GENERATOR] Creating high-fidelity sample leaf images in: {output_dir}")

    # 1. Potato Early Blight
    c1 = create_base_canvas()
    m1 = draw_potato_leaf_shape(c1)
    add_early_blight_bullseyes(c1, m1, [
        (220, 220, 26), (290, 270, 32), (190, 310, 22), (270, 170, 18), (230, 360, 20)
    ])
    cv2.imwrite(os.path.join(output_dir, "potato_early_blight.jpg"), cv2.cvtColor(c1, cv2.COLOR_RGB2BGR))

    # 2. Potato Late Blight
    c2 = create_base_canvas()
    m2 = draw_potato_leaf_shape(c2)
    add_late_blight_lesions(c2, m2, [
        (180, 210, 50, 65), (320, 280, 60, 75), (250, 370, 45, 50)
    ])
    cv2.imwrite(os.path.join(output_dir, "potato_late_blight.jpg"), cv2.cvtColor(c2, cv2.COLOR_RGB2BGR))

    # 3. Potato Healthy
    c3 = create_base_canvas()
    draw_potato_leaf_shape(c3, green_color=(45, 155, 48))
    cv2.imwrite(os.path.join(output_dir, "potato_healthy.jpg"), cv2.cvtColor(c3, cv2.COLOR_RGB2BGR))

    # 4. Tomato Early Blight
    c4 = create_base_canvas()
    m4 = draw_tomato_leaf_shape(c4)
    add_early_blight_bullseyes(c4, m4, [
        (230, 210, 24), (280, 290, 28), (210, 330, 20), (290, 190, 16)
    ])
    cv2.imwrite(os.path.join(output_dir, "tomato_early_blight.jpg"), cv2.cvtColor(c4, cv2.COLOR_RGB2BGR))

    # 5. Tomato Bacterial Spot
    c5 = create_base_canvas()
    m5 = draw_tomato_leaf_shape(c5)
    add_bacterial_spots(c5, m5, num_spots=50)
    cv2.imwrite(os.path.join(output_dir, "tomato_bacterial_spot.jpg"), cv2.cvtColor(c5, cv2.COLOR_RGB2BGR))

    # 6. Tomato Healthy
    c6 = create_base_canvas()
    draw_tomato_leaf_shape(c6, green_color=(40, 150, 45))
    cv2.imwrite(os.path.join(output_dir, "tomato_healthy.jpg"), cv2.cvtColor(c6, cv2.COLOR_RGB2BGR))

    # 7. Pepper Bell Bacterial Spot
    c7 = create_base_canvas()
    m7 = draw_pepper_leaf_shape(c7)
    add_bacterial_spots(c7, m7, num_spots=40)
    cv2.imwrite(os.path.join(output_dir, "pepper_bacterial_spot.jpg"), cv2.cvtColor(c7, cv2.COLOR_RGB2BGR))

    # 8. Pepper Bell Healthy
    c8 = create_base_canvas()
    draw_pepper_leaf_shape(c8, green_color=(35, 140, 40))
    cv2.imwrite(os.path.join(output_dir, "pepper_healthy.jpg"), cv2.cvtColor(c8, cv2.COLOR_RGB2BGR))

    print(f"[GENERATOR] Successfully created 8 sample leaf images in '{output_dir}'.")


if __name__ == "__main__":
    generate_all_samples()
