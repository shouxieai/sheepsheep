import sheepsheep
import cv2
import numpy as np

env = sheepsheep.Environment(stack_use_plane=False)
state = env.reset()
action = np.random.choice(np.where(state[1].reshape(-1) == 2)[0])

while True:
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
    image = env.render()

    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if done:
        print("Game over")
        state = env.get_env_state()
        action = np.random.choice(np.where(state[1].reshape(-1) == 2)[0])
        continue

    action = np.random.choice(np.where(state[1].reshape(-1) == 2)[0])

cv2.destroyAllWindows()