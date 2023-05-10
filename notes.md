## scripts
```powershell
# sync upstream
git fetch upstream
git stash
git rebase upstream/master
git stash apply

# diff homework
tree homework /f
git diff --no-index examples/gltfloading/gltfloading.cpp homework/homework1/homework1.cpp
git diff --no-index examples/gltfskinning/gltfskinning.cpp homework/homework1/homework1.cpp
```

## homework1
refs: pbrbasic, gltfskinning, gltfloading, pbribl
```powershell
cd D:\Projects\games106\data\homework\shaders\glsl\homework1
glslc -c mesh.vert;glslc -c mesh.frag;
```