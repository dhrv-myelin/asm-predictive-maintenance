"""add statistical_patterns table

Revision ID: a1b2c3d4e5f6
Revises: e87fa6a3d109
Create Date: 2026-03-23

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'e87fa6a3d109'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'statistical_patterns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('statistical_pattern', sa.Text(), nullable=True),
        sa.Column('likely_causes_and_what_to_inspect', sa.Text(), nullable=True),
        sa.Column('metric_name', sa.Text(), nullable=True),
        sa.Column('type_of_motor', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('statistical_patterns')