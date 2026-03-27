"""add columns

Revision ID: e87fa6a3d109
Revises: 3952f744f16c
Create Date: 2026-03-21
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'e87fa6a3d109'
down_revision: Union[str, None] = '3952f744f16c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # process_metrics
    op.add_column(
        'process_metrics',
        sa.Column('cycle_count', sa.Float(), nullable=True)
    )
    op.add_column(
        'process_metrics',
        sa.Column('pallet_serial_number', sa.String(), nullable=True)
    )

    # model_predictions
    op.add_column(
        'model_predictions',
        sa.Column('patterns', sa.Text(), nullable=True)
    )


def downgrade() -> None:
    # reverse in opposite order

    op.execute('ALTER TABLE model_predictions DROP COLUMN IF EXISTS patterns;') #wasn't working for op.drop_column had to use op.execute with sql query
    op.execute('ALTER TABLE process_metrics DROP COLUMN IF EXISTS pallet_serial_number;')
    op.execute('ALTER TABLE process_metrics DROP COLUMN IF EXISTS cycle_count;')