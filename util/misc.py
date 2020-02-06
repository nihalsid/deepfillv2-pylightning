import random


def generate_random_box(image_size, bbox_shape, bbox_randomness, bbox_margin):
    bbox_height = random.randint(int(bbox_shape * (1 - bbox_randomness)), int(bbox_shape * (1 + bbox_randomness)))
    bbox_width = random.randint(int(bbox_shape * (1 - bbox_randomness)), int(bbox_shape * (1 + bbox_randomness)))
    y_min = random.randint(bbox_margin, image_size - bbox_height - bbox_margin - 1)
    x_min = random.randint(bbox_margin, image_size - bbox_width - bbox_margin - 1)
    return y_min, x_min, bbox_height, bbox_width


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))
    print(generate_random_box(256, 48, 0.25, 32))

