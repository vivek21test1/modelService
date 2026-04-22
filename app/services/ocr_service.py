import base64
import io
import logging
import time

import numpy as np
import easyocr
from PIL import Image

from ..schemas.ocr import OCRResponse

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, reader: easyocr.Reader) -> None:
        self._reader = reader
        logger.info("OCRService initialised with reader: %s", type(reader).__name__)

    def process_bytes(self, image_bytes: bytes) -> OCRResponse:
        image_size_kb = len(image_bytes) / 1024
        logger.info("Received image — size: %.1f KB", image_size_kb)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        logger.info("Image decoded — dimensions: %dx%d px", width, height)

        image_np = np.array(image)

        logger.info("Starting EasyOCR inference ...")
        t_start = time.perf_counter()
        results = self._reader.readtext(image_np)
        elapsed = time.perf_counter() - t_start

        lines = [text for _, text, _ in results]
        word_count = sum(len(line.split()) for line in lines)

        logger.info(
            "OCR complete — %.2fs | %d region(s) detected | %d word(s) extracted",
            elapsed, len(results), word_count,
        )
        return OCRResponse(text=" ".join(lines), lines=lines)

    def process_base64(self, image_base64: str) -> OCRResponse:
        logger.info("Decoding base64 image ...")
        image_bytes = base64.b64decode(image_base64)
        logger.info("Base64 decoded — raw size: %.1f KB", len(image_bytes) / 1024)
        return self.process_bytes(image_bytes)
