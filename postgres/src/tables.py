from sqlalchemy import Column, Integer, String, Float, DateTime, func, UniqueConstraint
from sqlalchemy.orm import declarative_base

# Base = declarative_base()

# class SkillRelation(Base):
#     __tablename__ = "skill_relations"
#     id = Column(Integer, primary_key=True)
#     base_skill = Column(String, index=True)
#     related_skill = Column(String, index=True)
#     similarity = Column(Float)
#     source = Column(String, default="embedding")
#     created_at = Column(DateTime, server_default=func.now())

#     __table_args__ = (UniqueConstraint('base_skill', 'related_skill', name='uq_base_related'),)

# class Skill(Base): 
#     __tablename__ = "skills"
#     id = Column(Integer, primary_key=True)
#     skill = Column(String, index=True)

