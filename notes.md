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

cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -SD:/Projects/games106 -Bd:/Projects/games106/build -G "Visual Studio 17 2022" -T host=x64 -A x64
```

## homework
```powershell
# glsl compile
cd D:\Projects\games106\data\homework\shaders\glsl\homework1
glslc -c mesh.vert;glslc -c mesh.frag;

# compress
7z a -t7z homework1.7z homework/homework1/homework1.cpp data/homework/shaders/glsl/homework1/*

```