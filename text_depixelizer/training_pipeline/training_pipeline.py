import logging
from random import randint
import time
from typing import List, Tuple
from nltk.corpus import brown

from PIL import Image
from PIL.ImageFont import FreeTypeFont

from text_depixelizer.parameters import PictureParameters
from text_depixelizer.training_pipeline.original_image import ImageCreationOptions, OriginalImage, generate_image_from_text
from text_depixelizer.training_pipeline.pixelized_image import PixelizationOptions, PixelizedImage, pixelize_image
from text_depixelizer.training_pipeline.windows import WindowOptions, Window, create_windows_from_image
from text_depixelizer.training_pipeline.text_generator import RegexTextGenerator
import matplotlib.pyplot as plt
import numpy as np


def create_training_data(n_img: int, picture_parameters: PictureParameters) \
        -> Tuple[List[str], List[OriginalImage], List[PixelizedImage], List[List[Window]]]:
    """
    Generates the data required for training the HMM.生成训练数据
    """

    texts: List[str] = generate_texts(n_img, picture_parameters.pattern)
    # texts[0] = "englishstring"
    # 改动：修改text 换成真实数据
    # texts: List[str] = brown.words()[:n_img]

    original_images: List[OriginalImage] = generate_original_images(  # 生成原始图片
        texts=texts,
        font=picture_parameters.font,
        font_color=picture_parameters.font_color,
        background_color=picture_parameters.background_color
    )
    # 保存第一张原始图片
    print("保存第一张原始图片")
    first_original_image = original_images[0].img
    first_original_image.save("/Users/lianghaiyan/Desktop/pic/original.png")
    #
    # # 显示图像
    # plt.imshow(first_original_image)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()

    pixelized_images: List[PixelizedImage] = generate_pixelized_images(  # 生成像素图片
        original_images,
        picture_parameters.block_size,
        picture_parameters.randomize_pixelization_origin_x,
        picture_parameters.offset_y
    )

    # 保存第一张像素化图片
    print("保存第一张像素化图片")
    pixelized_image = pixelized_images[0].image
    pixelized_image.save("/Users/lianghaiyan/Desktop/pic/output.png")






    windows: List[List[Window]] = generate_windows(original_images, pixelized_images, picture_parameters.window_size)
    return texts, original_images, pixelized_images, windows


def generate_texts(n_img: int, pattern: str) -> List[str]:
    """
    Generates n_img strings that follow the given regex pattern
    """

    time_logger: logging.Logger = logging.getLogger('time_logger')
    t: float = time.perf_counter()
    text_generator: RegexTextGenerator = RegexTextGenerator(pattern=pattern)
    texts: List[str] = [text_generator.generate_text() for _ in range(n_img)]
    print(n_img)
    # texts: List[str] = brown.words()[:10000]

    if n_img > 100:
        time_logger.info(f'Created texts in {time.perf_counter() - t} seconds')
    print("输出文本", texts)
    return texts


def generate_original_images(
        texts: List[str],
        font: FreeTypeFont,
        font_color: Tuple[int, int, int] = (0, 0, 0),
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> List[OriginalImage]:
    """
    Given a list of texts and a font, generate images with that text and font
    Padding will be added around the text to allow space for pixelization that extends over the text's bounding box
    """
    time_logger: logging.Logger = logging.getLogger('time_logger')
    t = time.perf_counter()

    image_creation_options: ImageCreationOptions = ImageCreationOptions(
        padding=(20, 20),  # 修改padding
        font=font,
        font_color=font_color,
        background_color=background_color
    )

    original_images: List[OriginalImage] = [generate_image_from_text(text, image_creation_options) for text in texts]

    if len(texts) > 100:
        time_logger.info(f'Created original images in {time.perf_counter() - t} seconds')
    return original_images


def generate_pixelized_images(original_images: List[OriginalImage],
                              block_size: int,
                              randomize_pixelization_origin_x: bool,
                              pixelization_offset_y: int) -> List[PixelizedImage]:
    """
    Pixelizes the original images with the given block_size.  打码
    By default, the pixelization is in line with the baseline of the text and the right edge of the bounding box. This
    can be varied with the other two parameters
    """
    time_logger: logging.Logger = logging.getLogger('time_logger')
    t = time.perf_counter()

    pixelization_options: List[PixelizationOptions] = [PixelizationOptions(
        block_size,
        offset=(
            randint(0, block_size) if randomize_pixelization_origin_x else 0,  # 随机生成偏移量（在块大小内）
            pixelization_offset_y
        )
    ) for _ in range(len(original_images))]
    pixelized_images: List[PixelizedImage] = [pixelize_image(original_image, pix_o) for original_image, pix_o in zip(original_images, pixelization_options)]

    if len(original_images) > 100:
        time_logger.info(f'Pixelated images in {time.perf_counter() - t} seconds')

    return pixelized_images


def generate_windows(original_images: List[OriginalImage], pixelized_images: List[PixelizedImage], window_size: int) -> List[List[Window]]:
    """
    Generates the windows from the pixelized images.  # 生成窗口
    Note: The information from the original images is also needed, since we need to infer the characters that are in this window
    """
    time_logger: logging.Logger = logging.getLogger('time_logger')
    t = time.perf_counter()

    window_options: WindowOptions = WindowOptions(window_size=window_size, character_threshold=0)
    windows: List[List[Window]] = [
        create_windows_from_image(original_image, pixelized_image, window_options)
        for original_image, pixelized_image
        in zip(original_images, pixelized_images)
    ]
    if len(original_images) > 100:
        time_logger.info(f'Created windows in {time.perf_counter() - t} seconds')

    return windows
