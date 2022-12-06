from style_transfer import StyleTransfer

if __name__ == "__main__":
    style_transfer = StyleTransfer(
        input_image='data/input/1.jpg',
        example_image='data/example/1.jpg',
        input_mask='data/mask/1.jpg',
        example_mask='data/mask/1.jpg'
    )

    style_transfer.transfer()
