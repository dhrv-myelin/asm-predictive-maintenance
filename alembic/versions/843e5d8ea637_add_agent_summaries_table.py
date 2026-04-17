"""add agent summaries table

Revision ID: 843e5d8ea637
Revises: a1b2c3d4e5f6
Create Date: 2026-04-11
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = '843e5d8ea637'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'   # <-- updated
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'agent_summaries',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('run_date', sa.Date(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS agent_summaries;')