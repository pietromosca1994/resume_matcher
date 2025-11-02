#%%
# import modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.')))

from src.configs import ResumeProcessorConfig 
from src.resume_processor import ResumeProcessor 

#%%
# initialize Resume Processor
resume_processor_config=ResumeProcessorConfig()
resume_processor=ResumeProcessor(resume_processor_config)

#%%
# run skills semantic
skills=["JavaScript", "Python", "Java"]
semantic_skills=resume_processor._expand_semantic(skills, 'skills_index', k=3)
print(semantic_skills)
# %%
