"""
Cloudinary Uploader - Upload images to Cloudinary CDN.

Works INDEPENDENTLY of PDF processing and embedding.
Use it if you just need to upload images to a CDN.

Features:
- Retry logic with timeouts
- Batch uploading
- Automatic JPEG optimization

Environment Variables:
- VISUAL_RAG_THREAD_SAFE: Set to "1" to use thread-safe timeouts
  (required for Streamlit, Flask, or other threaded contexts)
"""

import io
import logging
import os
import platform
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

THREAD_SAFE_MODE = os.getenv("VISUAL_RAG_THREAD_SAFE", "").lower() in ("1", "true", "yes")


class CloudinaryUploader:
    """
    Upload images to Cloudinary CDN.

    Works independently - just needs PIL images.

    Args:
        cloud_name: Cloudinary cloud name
        api_key: Cloudinary API key
        api_secret: Cloudinary API secret
        folder: Base folder for uploads
        max_retries: Number of retry attempts
        timeout_seconds: Timeout per upload

    Example:
        >>> uploader = CloudinaryUploader(
        ...     cloud_name="my-cloud",
        ...     api_key="xxx",
        ...     api_secret="yyy",
        ...     folder="my-project",
        ... )
        >>>
        >>> url = uploader.upload(image, "doc_page_1")
        >>> print(url)  # https://res.cloudinary.com/.../doc_page_1.jpg
    """

    def __init__(
        self,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        folder: str = "visual-rag",
        max_retries: int = 3,
        timeout_seconds: int = 30,
        jpeg_quality: int = 95,
    ):
        # Load from environment if not provided
        self.cloud_name = cloud_name or os.getenv("CLOUDINARY_CLOUD_NAME")
        self.api_key = api_key or os.getenv("CLOUDINARY_API_KEY")
        self.api_secret = api_secret or os.getenv("CLOUDINARY_API_SECRET")

        if not all([self.cloud_name, self.api_key, self.api_secret]):
            raise ValueError(
                "Cloudinary credentials required. Set CLOUDINARY_CLOUD_NAME, "
                "CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET environment variables "
                "or pass them as arguments."
            )

        self.folder = folder
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.jpeg_quality = jpeg_quality

        # Check dependency
        try:
            import cloudinary  # noqa
        except ImportError:
            raise ImportError(
                "Cloudinary not installed. "
                "Install with: pip install visual-rag-toolkit[cloudinary]"
            )

        logger.info("☁️ Cloudinary uploader initialized")
        logger.info(f"   Folder: {folder}")

    def upload(
        self,
        image: Image.Image,
        public_id: str,
        subfolder: Optional[str] = None,
    ) -> Optional[str]:
        """
        Upload a single image to Cloudinary.

        Args:
            image: PIL Image to upload
            public_id: Public ID (filename without extension)
            subfolder: Optional subfolder within base folder

        Returns:
            Secure URL of uploaded image, or None if failed
        """
        import cloudinary
        import cloudinary.uploader

        # Prepare buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)

        # Configure Cloudinary
        cloudinary.config(
            cloud_name=self.cloud_name,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

        # Build folder path
        folder_path = self.folder
        if subfolder:
            folder_path = f"{self.folder}/{subfolder}"

        def do_upload():
            buffer.seek(0)
            result = cloudinary.uploader.upload(
                buffer,
                folder=folder_path,
                overwrite=True,
                public_id=public_id,
                resource_type="image",
                timeout=self.timeout_seconds,
            )
            return result["secure_url"]

        # Use thread-safe mode for Streamlit/Flask/threaded contexts
        # Set VISUAL_RAG_THREAD_SAFE=1 to enable
        if THREAD_SAFE_MODE or threading.current_thread() is not threading.main_thread():
            return self._upload_with_thread_timeout(do_upload, public_id)
        else:
            return self._upload_with_signal_timeout(do_upload, public_id)

    def _upload_with_thread_timeout(self, do_upload, public_id: str) -> Optional[str]:
        """Thread-safe upload with ThreadPoolExecutor timeout."""
        for attempt in range(self.max_retries):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(do_upload)
                    return future.result(timeout=self.timeout_seconds)

            except FuturesTimeoutError:
                logger.warning(
                    f"Upload timeout (attempt {attempt + 1}/{self.max_retries}): {public_id}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

            except Exception as e:
                logger.warning(f"Upload failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        logger.error(f"❌ Upload failed after {self.max_retries} attempts: {public_id}")
        return None

    def _upload_with_signal_timeout(self, do_upload, public_id: str) -> Optional[str]:
        """Signal-based upload timeout (main thread only, Unix/macOS)."""
        use_timeout = platform.system() != "Windows"

        class SignalTimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise SignalTimeoutError(f"Upload timed out after {self.timeout_seconds}s")

        for attempt in range(self.max_retries):
            try:
                if use_timeout:
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.timeout_seconds)

                try:
                    return do_upload()
                finally:
                    if use_timeout:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)

            except SignalTimeoutError:
                logger.warning(
                    f"Upload timeout (attempt {attempt + 1}/{self.max_retries}): {public_id}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

            except Exception as e:
                logger.warning(f"Upload failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        logger.error(f"❌ Upload failed after {self.max_retries} attempts: {public_id}")
        return None

    def upload_original_and_resized(
        self,
        original_image: Image.Image,
        resized_image: Image.Image,
        base_public_id: str,
    ) -> tuple:
        """
        Upload both original and resized versions.

        Args:
            original_image: Original PDF page image
            resized_image: Resized image for ColPali
            base_public_id: Base public ID (e.g., "doc_page_1")

        Returns:
            Tuple of (original_url, resized_url) - either can be None on failure
        """
        original_url = self.upload(
            original_image,
            base_public_id,
            subfolder="original",
        )

        resized_url = self.upload(
            resized_image,
            base_public_id,
            subfolder="resized",
        )

        return original_url, resized_url

    def upload_original_cropped_and_resized(
        self,
        original_image: Image.Image,
        cropped_image: Optional[Image.Image],
        resized_image: Image.Image,
        base_public_id: str,
    ) -> tuple:
        original_url = self.upload(
            original_image,
            base_public_id,
            subfolder="original",
        )

        cropped_url = None
        if cropped_image is not None:
            cropped_url = self.upload(
                cropped_image,
                base_public_id,
                subfolder="cropped",
            )

        resized_url = self.upload(
            resized_image,
            base_public_id,
            subfolder="resized",
        )

        return original_url, cropped_url, resized_url
