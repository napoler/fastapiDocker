# FastApi Docker



## 构建命令

```
# 构建
docker build -t myimage .

# 启动容器

docker run -d --name mycontainer -p 80:80 myimage
# docker run -d -p 8094:80 fastapiner
```
docker save fastapiner:latest > ../fastapinerv01.tar

docker load < ../fastapinerv01.tar



更多文档参考 https://fastapi.tiangolo.com/deployment/docker/
https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker

Make Jelly site have a GitBook look!

## Demo

[https://www.terrychan.org/jekyll-gitbook/](https://www.terrychan.org/jekyll-gitbook/)

## Why Jekyll with GitBook

GitBook is an amazing frontend style to present and organize contents (such as book chapters
and blogs) on Web. The typical to deploy GitBook at [Github Pages][1]
is building HTML files locally and then push to Github repository, usually to the `gh-pages`
branch. It's quite annoying to repeat such workload and make it hard for people do version
control via git for when there are generated HTML files to be staged in and out.

This theme takes style definition out of generated GitBook site and provided the template
for Jekyll to rendering markdown documents to HTML, thus the whole site can be deployed
to [Github Pages][1] without generating and uploading HTML bundle every time when there are
changes to the original repo.

## How to Get Started

This theme can be used just as other [Jekyll themes][1].

[Fork][3] this repository and add your markdown posts to the `_posts` folder.

## License

This work is open sourced under the Apache License, Version 2.0.

Copyright 2019 Tao He.

[1]: https://pages.github.com
[2]: https://pages.github.com/themes
[3]: https://github.com/sighingnow/jekyll-gitbook/fork