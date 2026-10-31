"""Allow the frontend to PUT presigned uploads straight to the bucket.

A browser PUT to another origin (the bucket) is a CORS request with a
preflight; without this rule the browser blocks every direct upload before
it is sent. One-time setup per bucket, using the same OBJECT_STORAGE_*
settings as the API. Works on Cloudflare R2 and AWS S3.

    python scripts/configure_bucket_cors.py --origin http://localhost:3000 \
        --origin https://app.example.com
"""

from __future__ import annotations

import click

from app.core.config import get_settings
from app.storage import s3_client


@click.command()
@click.option("--origin", "origins", multiple=True, required=True, help="Frontend origin.")
def main(origins: tuple[str, ...]) -> None:
    bucket = get_settings().object_storage_bucket
    if not bucket:
        raise click.ClickException("OBJECT_STORAGE_BUCKET is not set")
    s3_client().put_bucket_cors(
        Bucket=bucket,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedOrigins": list(origins),
                    "AllowedMethods": ["PUT"],
                    "AllowedHeaders": ["content-type"],
                    "MaxAgeSeconds": 3600,
                }
            ]
        },
    )
    click.echo(f"CORS set on {bucket} for {', '.join(origins)}")


if __name__ == "__main__":
    main()
