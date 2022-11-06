import sheepsheep
import cv2
import numpy as np

env = sheepsheep.Environment(stack_use_plane=False)
state = env.reset()
action = np.where(state[0].reshape(-1) != 0)[0][0]
niter = 0

while True:

    niter += 1
    print(f"Action is {action}")

    # action -> [0, 334)
    # body[18x18] + external[3] + stack[4] + button[3]
    state, reward, done = env.step_serial(action)

    # 主体部分，      主体mask，     底下已经选择的   移动出来的        堆叠的部分     功能按钮
    # body_elements(18x18 sub-block) -> (9x9 block).  0 0 0 0 = empty, 4 5 6 7 = element1, 8 9 10 11 = element2
    #            max-value is (num_element + 1) * 4
    # body_mask_flag(18x18 sub-block) -> (9x9 block).  0 is empty, 1 is masked, 2 is unmasked
    #            element unmasked if that in top level
    #            element masked   if that in non-top level
    #            element empty    if that is no element
    # bar_state(7) list      ->  container of already select element
    # external_state(3) list ->  container of move to upper element by MoveTo props
    # stack_state(4) list    ->  container of top stack element
    # button_state(3) list   ->  state of props(num_shuffle, num_push, num_back)
    body_elements, body_mask_flag, bar_state, external_state, stack_state, button_state = state
    y, x = np.where(body_mask_flag == 2)
    v = body_elements[y, x] // 4

    image = env.render()
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if done:
        print(f"Steps: {niter}")
        state = env.get_env_state()
        action = np.where(state[0].reshape(-1) != 0)[0][0]
        continue

    simple_state = np.concatenate([v, external_state, stack_state])
    if len(bar_state) == 0:
        action = np.where(simple_state != 0)[0][0]
    else:
        action = None
        counter = [(item, bar_state.count(item)) for item in set(bar_state) if item != 0]
        if len(counter) > 0:
            counter = sorted(counter, key=lambda x: x[1], reverse=True)
            for item, count in counter:

                toffset = np.where(simple_state == item)[0]
                if len(toffset) > 0:
                    action = toffset[0]
                    break

            if action is None:
                counter_unclick = [(item, list(simple_state).count(item)) for item in set(simple_state) if item != 0]
                counter_unclick = sorted(counter_unclick, key=lambda x: x[1], reverse=True)
                item, count = counter_unclick[0]
                action = np.where(simple_state == item)[0][0]

        if action is None:
            action = np.where(simple_state != 0)[0][0]

    if action < len(v):
        action = y[action] * body_elements.shape[1] + x[action]
    else:
        action = action - len(v) + body_elements.shape[0] * body_elements.shape[1]


cv2.destroyAllWindows()