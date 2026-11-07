"""Render collision-checked GIFs with the demo renderer and real training snapshots.

The capture executable is built temporarily. No playback/control interface is added
 to the interactive game. Requires Pillow, NumPy, CMake build dependencies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MAP = ["############", "#..........#", "#..####....#", "#..#.......#",
       "#..#..##...#", "#.....##...#", "#........#.#", "#..........#", "############"]
BALL = np.array([5.5, .5, 6.5])
RADIUS = .18
EXPOSURE = .12
SIZE = 256


def clearance(p):
    x, y, z = p
    if not (RADIUS <= y <= 1 - RADIUS):
        raise ValueError("Camera intersects floor/ceiling")
    distances = []
    for r, row in enumerate(MAP):
        for c, tile in enumerate(row):
            if tile == "#":
                distances.append(math.hypot(max(c-x, 0, x-c-1), max(r-z, 0, z-r-1)))
    # The analytic silver box and copper sphere are solid obstacles.
    distances.append(float(np.linalg.norm(np.maximum(np.maximum(
        np.array([9., 0., 4.])-p, p-np.array([9.4, 1., 4.4])), 0))))
    distances.append(float(np.linalg.norm(p-BALL))-.45)
    d = min(distances)
    if d < RADIUS - 1e-6:
        raise ValueError(f"Camera collision at {p}: clearance={d}")
    return d


def make_tour():
    points = []
    segments = [([2.5,.5,5.5], [2.5,.5,6.5], 6),
                ([2.5,.5,6.5], [4.25,.5,6.5], 8)]
    for a,b,n in segments:
        for t in np.linspace(0,1,n,endpoint=False):
            points.append(np.array(a)*(1-t)+np.array(b)*t)
    # Sweep around the near half of the mirror ball, staying in the lower corridor.
    for angle in np.linspace(math.pi,0,24):
        points.append(BALL+np.array([1.25*math.cos(angle),0,1.25*math.sin(angle)]))
    points += points[-2:0:-1]
    # Check the swept path every .01 units, not just the rendered camera samples.
    minimum = 1e9
    for a,b in zip(points, points[1:]+points[:1]):
        for t in np.linspace(0,1,max(2,math.ceil(np.linalg.norm(b-a)/.01)+1)):
            minimum=min(minimum,clearance(a*(1-t)+b*t))
    return points,minimum


def pose(p):
    delta=BALL-p
    # Fixed light, camera, exposure, and analytic geometry across training checkpoints.
    return [*map(float,p), math.atan2(delta[2],delta[0]), 0., 3.0, .5, 7.1]


def build_capture(dest):
    source=(ROOT/'src/main.cpp').read_text()
    source='#include <array>\n#include <cstdlib>\n'+source
    a=source.index('  if (!headless) SDL_SetRelativeMouseMode')
    source=source[:a]+\
        '  std::ifstream cp(std::getenv("DEMO_CAMERA_PATH"));\n'\
        '  std::vector<std::array<float, 8>> poses; std::array<float, 8> v;\n'\
        '  while(cp >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5] >> v[6] >> v[7]) poses.push_back(v);\n'\
        '  if (poses.empty()) return 2;\n'+source[a:]
    a=source.index('    // -------- input / camera update')
    b=source.index('    // -------- pass 1:',a)
    source=source[:a]+\
        '    const auto &p = poses[std::min(frame, (int)poses.size()-1)];\n'\
        '    cam.pos={p[0],p[1],p[2]}; cam.yaw=p[3]; cam.pitch=p[4];\n'\
        '    cam.torch_follow=false; cam.torch_pos={p[5],p[6],p[7]};\n'+source[b:]
    a=source.index('    // -------- HUD (burned into the frame)')
    b=source.index('    // -------- optional screenshot',a)
    source=source[:a]+source[b:]
    source=source.replace('"uExposure"), 1.0f)',f'"uExposure"), {EXPOSURE}f)')
    (dest/'capture.cpp').write_text(source)
    flags=subprocess.check_output(['pkg-config','--cflags','--libs','sdl2'],text=True).split()
    binary=dest/'capture'
    subprocess.run(['g++','-O2','-fopenmp','-std=c++17','-I'+str(ROOT/'src'),
                    str(dest/'capture.cpp'),'-o',str(binary),*flags,'-lGL'],check=True)
    return binary


def render(binary, work, model, poses):
    work.mkdir(parents=True,exist_ok=True)
    camera=work/'poses.txt'
    camera.write_text(''.join(' '.join(map(str,pose(p)))+'\n' for p in poses))
    env=dict(os.environ,DEMO_CAMERA_PATH=str(camera),OMP_NUM_THREADS='2')
    subprocess.run([str(binary),'--model',str(model),'--headless','--res',str(SIZE),
                    '--lights','3','--frames',str(len(poses)),'--save',str(work)],
                   cwd=ROOT,env=env,check=True)
    return [Image.open(f).convert('RGB') for f in sorted(work.glob('frame_*.ppm'))]


def font(size):
    return ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',size) if Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf').exists() else ImageFont.load_default()


def step_count(im, step, total):
    out=im.copy()
    ImageDraw.Draw(out).text((5,4),f"{step} / {total}",font=font(12),fill=(255,230,0))
    return out


def gif(frames,path,durations):
    # One palette across the entire animation prevents per-frame color pumping.
    sample=Image.new('RGB',(frames[0].width,frames[0].height*min(12,len(frames))))
    for j,i in enumerate(np.linspace(0,len(frames)-1,min(12,len(frames))).astype(int)):
        sample.paste(frames[i],(0,j*frames[0].height))
    palette=sample.quantize(colors=256)
    q=[f.quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    q[0].save(path,save_all=True,append_images=q[1:],duration=durations,loop=0,optimize=False,disposal=2)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default='learning_gif')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--tour-only',action='store_true')
    mode.add_argument('--training-only',action='store_true')
    args=ap.parse_args()
    out=ROOT/'screenshots'; out.mkdir(exist_ok=True)
    points,min_clearance=make_tour()
    with tempfile.TemporaryDirectory(prefix='hashgrid-media-') as tmp:
        tmp=Path(tmp); binary=build_capture(tmp)
        if args.training_only:
            manifest=json.loads((out/'render-manifest.json').read_text())
        else:
            images=render(binary,tmp/'tour',ROOT/'model',points)
            frames=images
            gif(frames,out/'environment.gif',100)
            images[len(images)//3].save(out/'in-game.png')
            manifest={'tour':{'frames':len(images),'fps':10,'camera_radius':RADIUS,
                'minimum_obstacle_clearance':min_clearance,'mirror_ball_center':BALL.tolist(),'model_source':'Bundled verified model in model/','camera_positions':[p.tolist() for p in points],
                'swept_path_sample_spacing':.01},'render':{'exposure':EXPOSURE,
                'lights':3,'torch_position':[3.0,.5,7.1],'resolution':SIZE,'note':'Same analytic game geometry and GLSL lighting renderer. Fixed off-camera torch for a clear view.'}}
        if not args.tour_only:
            run=ROOT/'training/runs'/args.run
            steps=sorted((run/'progress').glob('step_*'))
            if not steps or int(steps[0].name.split('_')[1])!=0:
                raise ValueError('Missing initialized model snapshot')
            final_arch=json.loads((run/'model/model.json').read_text())
            final_step=int(steps[-1].name.split('_')[1])
            if final_step != final_arch['iters']:
                raise ValueError('Training must finish before rendering the progression')
            with np.load(run/'model/model.npz') as saved, np.load(steps[-1]/'model/model.npz') as snap:
                if any(not np.array_equal(saved[k],snap[k]) for k in saved.files):
                    raise ValueError('Final snapshot does not match the completed trained model')
            losses=json.loads((run/'progress/loss.json').read_text())
            lossmap={row['step']:row['batch_loss'] for row in losses}
            from stage4_export.export import export
            for step in steps:
                export(str(step.relative_to(ROOT/'training/runs')))
            viewpoint=np.array([4.25,.5,7.35]); clearance(viewpoint)
            frames=[]; total=int(steps[-1].name.split('_')[1]); evidence=[]
            for i,step in enumerate(steps):
                iteration=int(step.name.split('_')[1])
                image=render(binary,tmp/f'step{i}',step/'export',[viewpoint])[0]
                frames.append(step_count(image,iteration,total))
                evidence.append({'step':iteration,'batch_loss':lossmap.get(iteration),'model_sha256':hashlib.sha256((step/'model/model.npz').read_bytes()).hexdigest()})
            gif(frames,out/'training.gif',[1000]+[240]*(len(frames)-2)+[2200])
            manifest['training']={'run':args.run,'snapshots':evidence,'fixed_camera':viewpoint.tolist(),
                'display':'Current model only; yellow optimizer-step count in the top-left.',
                'geometry':'Fixed analytic geometry/reflections; only hashgrid+MLP lighting is learned.', 'completed_training_matches_last_snapshot': True}
        (out/'render-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    main()
