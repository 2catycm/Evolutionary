#%%
from pathlib import Path
this_file = Path(__file__)
this_directory = this_file.parent
matlab_project = this_directory.parent/'Assignment1_matlab'
matlab_project
#%%
import matlabparser as mpars

fail_list = []
matlabs = matlab_project.rglob('*.m')
for m_files in matlabs:
    
    rel = m_files.relative_to(matlab_project)
    print(f"正在生成{rel}")
    # print(rel.stem)
    # print(rel.parent)
    target_dir = this_directory/rel.parent
    target_dir.mkdir(exist_ok=True)
    try:
        pylines = mpars.matlab2python(m_files.as_posix(), output=(target_dir/f"{rel.stem}.py").as_posix())
    except Exception as e:
        print(f"生成{rel}失败")
        print(e)
        # content = open(m_files).read()
        fail_list.append(rel)
        
        continue
    
# %%
fail_list
# %%
