import tensorflow as tf
from tensorflow.keras import layers, Model


def get_encoder(input_shape, latent_channels):
    """
    Encoder: maps concatenated input frames to a latent space.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(latent_channels, 3, strides=2, padding='same', activation='relu')(x)
    return Model(inputs, x, name="encoder")


def get_decoder(latent_shape, output_channels, img_height, img_width):
    """
    Decoder: reconstructs an image from the latent space.
    """
    latent_inputs = layers.Input(shape=latent_shape)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(latent_inputs)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='tanh')(x)
    output = layers.Cropping2D(cropping=((x.shape[1] - img_height, 0), (x.shape[2] - img_width, 0)))(x)
    return Model(latent_inputs, output, name="decoder")


def get_unet_block(input_shape, base_filters):
    """
    UNet-like block that performs denoising in latent space.
    Adjusted to ensure that concatenated tensors have matching spatial dimensions.
    """
    inputs = layers.Input(shape=input_shape)

    d1 = layers.Conv2D(base_filters, 3, strides=2, padding='same', activation='relu')(
        inputs)
    d2 = layers.Conv2D(base_filters * 2, 3, strides=2, padding='same', activation='relu')(
        d1)

    b = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')(d2)  # (9, 15, base_filters*4)

    u1 = layers.Concatenate()([b, d2])

    u2 = layers.Conv2DTranspose(base_filters * 2, 3, strides=2, padding='same', activation='relu')(u1)

    u2 = layers.Cropping2D(cropping=((u2.shape[1] - d1.shape[1], 0), (u2.shape[2] - d1.shape[2], 0)))(u2)
    u2 = layers.Concatenate()([u2, d1])

    u3 = layers.Conv2DTranspose(input_shape[-1], 3, strides=2, padding='same', activation='relu')(u2)

    u3 = layers.Cropping2D(cropping=((u3.shape[1] - input_shape[0], 0), (u3.shape[2] - input_shape[1], 0)))(u3)
    outputs = layers.Add()([u3, inputs])

    return Model(inputs, outputs, name="unet_block")


def create_diffusion_frame_interpolation_model(img_height, img_width, num_channels, latent_channels=256):
    """
    This model takes two input frames and a noise level value (to be used in a full diffusion process)
    and produces an interpolated frame.

    The pipeline is as follows:
      1. Concatenate the two frames and encode into a latent space.
      2. Inject a noise level signal into the latent representation.
      3. Pass through a UNet-style block to denoise/refine the latent code.
      4. Decode the latent representation back to image space.

    In practice, one would train this model as part of an iterative diffusion process.
    """

    frame1 = layers.Input(shape=(img_height, img_width, num_channels), name="frame1")
    frame2 = layers.Input(shape=(img_height, img_width, num_channels), name="frame2")
    noise_level = layers.Input(shape=(1,), name="noise_level")

    merged_frames = layers.Concatenate(axis=-1)([frame1, frame2])

    encoder = get_encoder(merged_frames.shape[1:], latent_channels)
    latent = encoder(merged_frames)

    latent_shape = encoder.output_shape[1:]  # (H, W, latent_channels)
    flat_dim = latent_shape[0] * latent_shape[1] * latent_shape[2]
    noise_proj = layers.Dense(flat_dim, activation='relu')(noise_level)
    noise_proj = layers.Reshape(latent_shape)(noise_proj)

    latent = layers.Add()([latent, noise_proj])

    unet = get_unet_block(latent_shape, base_filters=latent_channels // 8)
    latent_denoised = unet(latent)

    decoder = get_decoder(latent_denoised.shape[1:], num_channels, img_height, img_width)
    interpolated_frame = decoder(latent_denoised)

    return Model(inputs=[frame1, frame2, noise_level], outputs=interpolated_frame, name="DiffusionFrameInterpolation")


if __name__ == '__main__':
    model = create_diffusion_frame_interpolation_model(img_height=270, img_width=480, num_channels=3)
    model.summary()
