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
```