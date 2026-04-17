"""update agent_summaries: replace summary with action_items and machine_health

Revision ID: b7f3a9c12d45
Revises: 843e5d8ea637
Create Date: 2026-04-14
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'b7f3a9c12d45'
down_revision: Union[str, None] = '843e5d8ea637'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('ALTER TABLE agent_summaries DROP COLUMN IF EXISTS summary')
    op.execute('ALTER TABLE agent_summaries ADD COLUMN IF NOT EXISTS action_items TEXT')
    op.execute('ALTER TABLE agent_summaries ADD COLUMN IF NOT EXISTS machine_health TEXT')


def downgrade() -> None:
    op.drop_column('agent_summaries', 'machine_health')
    op.drop_column('agent_summaries', 'action_items')
    op.add_column('agent_summaries', sa.Column('summary', sa.Text(), nullable=True))