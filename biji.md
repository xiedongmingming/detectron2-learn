注册方式：
```
# 方式1
@BACKBONE_REGISTRY.register()
class MyBackbone():
	...
		
# 方式2
class MyBackbone():
	...
BACKBONE_REGISTRY.register(MyBackbone)
```