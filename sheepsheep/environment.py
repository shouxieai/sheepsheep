import cv2
import numpy as np
import pickle
import os

class Environment:
    def __init__(self, nlayer=10, num_block_per_layer=10, max_stack_num_block=10, seed=31, stack_use_plane=False):
        np.random.seed(seed)

        resource_file = os.path.join(os.path.dirname(__file__), "game.res")
        self.background, self.elements = pickle.load(open(resource_file, "rb"))
        self.bkg_color = 141, 255, 203
        self.element_size = 64
        self.max_num_stack = 4
        self.max_stack_num_block = max_stack_num_block
        self.max_num_bar   = 7
        self.max_num_external = 3
        self.origin_size = 72, 84
        self.num_block_per_layer = num_block_per_layer
        self.half_size = self.element_size // 2
        scale = self.element_size / self.origin_size[0]
        self.stack_use_plane = stack_use_plane

        for key in self.elements:
            element = self.elements[key]
            for i in range(len(element)):
                im = element[i]
                element[i] = cv2.resize(im, dsize=None, fx=scale, fy=scale)

        self.width, self.height = 10, 13
        self.imwidth = self.width * self.element_size
        self.imheight = self.height * self.element_size + int(self.origin_size[1] * scale) - self.element_size
        self.scene = np.full((self.imheight, self.imwidth, 4), self.bkg_color + (255,), dtype=np.uint8)

        if self.background.shape[0] > self.imheight or self.background.shape[1] > self.imwidth:
            scale = min(self.imheight / self.background.shape[0], self.imwidth / self.background.shape[1])
            self.background = cv2.resize(self.background, dsize=None, fx=scale, fy=scale)

        self.num_block = 16
        self.nlayer = nlayer
        self.body_width = 9
        self.body_height = 9
        self.reset()

    def rand_block(self):
        return np.random.randint(1, self.num_block + 1)
        # return np.random.randint(1, 5 + 1)

    def blending(self, dst, element, x, y):
        w, h = element.shape[1], element.shape[0]
        back_select = dst[y:y + h, x:x + w]
        back_rgb = back_select[..., :3]
        back_alpha = back_select[..., [3]] / 255.0
        fore_rgb = element[..., :3]
        fore_alpha = element[..., [3]] / 255.0

        merge_alpha = back_alpha * (1 - fore_alpha) + fore_alpha
        back_select[..., :3] = (
                    (back_rgb * back_alpha * (1 - fore_alpha) + fore_rgb * fore_alpha) / merge_alpha).astype(
            np.uint8)
        back_select[..., [3]] = (merge_alpha * 255).astype(np.uint8)

    def shuffle(self):
        # flag is [0 empty, 1 masked, 2 unmasked]
        self.layout = np.zeros((self.body_height * 2, self.body_width * 2, self.nlayer, 2), np.uint32)
        all_blocks = []
        all_blocks.extend(self.blocks)
        for items in self.stack_items:
            all_blocks.extend(items)

        np.random.shuffle(all_blocks)
        num_stack = len(self.stack_items)
        new_stack_items = [[] for i in range(num_stack)]
        self.blocks = []
        state = 0
        i = 0
        while i < len(all_blocks):
            item = all_blocks[i]
            if state < num_stack:
                if len(self.stack_items[state]) == 0:
                    state += 1
                    continue

                new_stack_items[state].append(item)
                if len(new_stack_items[state]) == len(self.stack_items[state]):
                    state += 1
            else:
                self.blocks.append(item)
            i += 1

        self.stack_items = new_stack_items

        i = 0
        while i < len(self.blocks):
            idd = self.blocks[i]
            x = np.random.randint(0, self.body_width)
            y = np.random.randint(0, self.body_height)
            ilayer = np.random.randint(0, self.nlayer)
            if self.add_body(idd, x, y, ilayer):
                i += 1

    def reset(self):

        self.num_shuffle = 1
        self.num_push = 1
        self.num_back = 1
        self.last_position = None
        self.last_value = None
        self.last_operator = None
        self.layout = np.zeros((self.body_height * 2, self.body_width * 2, self.nlayer, 2), np.uint32)
        num_body_block_pairs = self.nlayer * self.num_block_per_layer
        all_blocks = []
        for i in range(num_body_block_pairs):
            all_blocks.extend([self.rand_block()] * 3)

        self.finished_blocks = num_body_block_pairs * 3
        np.random.shuffle(all_blocks)
        num_stack = np.random.randint(2, self.max_num_stack + 1)
        stack_num_block = np.random.randint(self.max_stack_num_block // 2, self.max_stack_num_block + 1)
        self.stack_items = [[] for i in range(num_stack)]
        self.blocks = []
        state = 0
        for item in all_blocks:
            if state < num_stack:
                self.stack_items[state].append(item)
                if len(self.stack_items[state]) == stack_num_block:
                    state += 1
            else:
                self.blocks.append(item)

        i = 0
        while i < len(self.blocks):
            idd = self.blocks[i]
            x = np.random.randint(0, self.body_width)
            y = np.random.randint(0, self.body_height)
            ilayer = np.random.randint(0, self.nlayer)
            if self.add_body(idd, x, y, ilayer):
                i += 1

        self.bar_items = []
        self.extrnal_items = []
        self.update()
        return self.get_env_state()

    def render(self):
        self.scene[:] = self.bkg_color + (255,)
        x = (self.imwidth - self.background.shape[1]) // 2 // self.half_size
        y = (self.imheight - self.background.shape[0]) // 2 // self.half_size
        self.blending(self.scene, self.background, x, y)
        self.draw_body()
        for i, items in enumerate(self.stack_items):
            if len(items) > 0:
                self.draw_stack(items, i)

        for i, item in enumerate(self.extrnal_items):
            self.draw_external(item, i)

        for i, item in enumerate(self.bar_items):
            self.draw_bar(item, i)
        self.draw_spec_button()
        return self.scene

    def get_env_state(self):
        body_elements  = self.layout_squeeze[..., 1]
        body_mask_flag = self.layout_squeeze[..., 2]
        bar_state = self.bar_items + [0] * (self.max_num_bar - len(self.bar_items))
        external_state = self.extrnal_items + [0] * (self.max_num_external - len(self.extrnal_items))
        stack_state = [(self.stack_items[i][-1] if i < len(self.stack_items) and len(self.stack_items[i]) > 0 else 0) for i in range(self.max_num_stack)]
        button_state = self.num_shuffle, self.num_push, self.num_back
        return body_elements, body_mask_flag, bar_state, external_state, stack_state, button_state

    def update(self):
        for l in range(self.nlayer - 1, -1, -1):
            for ix in range(self.body_width):
                for iy in range(self.body_height):
                    x = ix * 2
                    y = iy * 2
                    if l % 2 == 1:
                        x += 1
                        y += 1
                        if ix == 8 or iy == 8: continue

                    # flag is [0 empty, 1 masked, 2 unmasked]
                    if self.layout[y, x, l, 0] == 0:
                        label = 0
                        flag = 0
                    else:
                        label = self.layout[y:y + 2, x:x + 2, l, 0]
                        masked = not np.bitwise_or(self.layout[y:y + 2, x:x + 2, l + 1:, 0] == label.reshape(2, 2, 1),
                                                   self.layout[y:y + 2, x:x + 2, l + 1:, 0] == 0).all()
                        flag = 1 if masked else 2
                    self.layout[y:y + 2, x:x + 2, l, 1] = flag

        # layer_index, idd, flag
        # flag is [0 empty, 1 masked, 2 unmasked]
        self.layout_squeeze = np.full((self.body_height * 2, self.body_width * 2, 3), (-1, 0, 0), dtype=np.int32)
        for y in range(self.body_height * 2):
            for x in range(self.body_width * 2):
                labeles = self.layout[y, x, :, 0]
                if labeles.sum() == 0:
                    label = 0
                    layer_idx = -1
                    flag = 0
                else:
                    labeles = labeles[::-1]
                    idx = np.where(np.bitwise_and(np.cumsum(labeles) == labeles, labeles != 0))[0][0]
                    label = labeles[idx]
                    layer_idx = self.nlayer - idx - 1
                    flag = self.layout[y, x, layer_idx, 1]
                self.layout_squeeze[y, x] = layer_idx, label, flag

    def draw_element(self, idd, x, y, mask=False, mask_use_m=False):
        image = self.elements[idd][1 if mask else 0]
        if mask and mask_use_m:
            image = self.elements["m"][0]
        self.blending(self.scene, image, int(x * self.half_size), int(y * self.half_size))

    def draw_body(self):
        for l in range(self.nlayer):
            for ix in range(self.body_width):
                for iy in range(self.body_height):
                    x = ix * 2
                    y = iy * 2
                    if l % 2 == 1:
                        x += 1
                        y += 1
                        if ix == 8 or iy == 8: continue

                    # flag is [0 empty, 1 masked, 2 unmasked]
                    if self.layout[y, x, l, 0] == 0:
                        continue

                    label, flag = self.layout[y:y + 2, x:x + 2, l][0, 0]
                    self.draw_element(label // 4, x + 1, y + 1, flag == 1)

    def add_body(self, idd, x, y, ilayer, mask=False):
        x = max(0, min(x, self.body_width - 1))
        y = max(0, min(y, self.body_height - 1))
        if ilayer % 2 == 1:
            x = max(0, min(x, self.body_width - 2))
            y = max(0, min(y, self.body_height - 2))

        x = x * 2
        y = y * 2
        if ilayer % 2 == 1:
            x += 1
            y += 1

        valid_position = self.layout[y:y + 2, x:x + 2, ilayer].sum() == 0
        if valid_position:
            self.layout[y:y + 2, x:x + 2, ilayer, 0] = idd * 4 + np.arange(4).reshape(2, 2)
        return valid_position

    def draw_bar(self, idd, x):
        x = max(0, min(x, 6)) * 2
        startx, starty = 1, 23
        self.draw_element(idd, startx + x, starty)

    def hit_button(self, x, y):
        x = int(x)
        y = int(y)
        if x >= 18 and x <= 19 and y >= 21 and y <= 22:
            return "shuffle" if self.num_shuffle > 0 else None
        elif x >= 16 and x <= 17 and y >= 22 and y <= 23:
            return "push" if self.num_push > 0 else None
        elif x >= 18 and x <= 19 and y >= 23 and y <= 24:
            return "back" if self.num_back > 0 else None

    def hit_external(self, x, y):
        x = int(x)
        y = int(y)
        if x >= 1 and x < min(self.max_num_external, len(self.extrnal_items)) * 2 + 1 and y >= 21 and y <= 22:
            return int(x - 1) // 2

    def hit_stack(self, x, y):
        startx, starty = 1, 19
        for i in range(len(self.stack_items)):
            if self.stack_use_plane:
                if len(self.stack_items[i]) == 0: continue
                tx, ty = (startx + i * 4), starty
                if x >= tx and x < tx + 2 and y >= ty and y < ty + 2:
                    return i
            else:
                if len(self.stack_items[i]) == 0: continue
                tx, ty = (startx + i * 4 + (len(self.stack_items[i]) - 1) * 0.2), starty
                if x >= tx and x < tx + 2 and y >= ty and y < ty + 2:
                    return i
        return -1

    def hit_body(self, x, y):
        x = int(x - 1)
        y = int(y - 1)
        if x < 0 or x >= self.body_width * 2 or y < 0 or y >= self.body_height * 2:
            return None

        # flag is [0 empty, 1 masked, 2 unmasked]
        flag = self.layout_squeeze[y, x, 2]
        if flag == 2:
            return x, y

    def draw_stack(self, idds, x):
        x = max(0, min(x, 4)) * 2
        startx, starty = 1, 19
        if self.stack_use_plane:
            self.draw_element(idds[-1], startx + x * 2, starty, False)
        else:
            for i, idd in enumerate(idds):
                self.draw_element(idd, startx + x * 2 + i * 0.2, starty, i < len(idds) - 1, True)

    def draw_external(self, idd, x):
        x = (x % self.max_num_external) * 2
        startx, starty = 1, 21
        self.draw_element(idd, startx + x, starty)

    def draw_spec_button(self):
        self.draw_element("shuffle", 18, 21, mask=self.num_shuffle < 1)
        self.draw_element("push", 16, 22, mask=self.num_push < 1)
        self.draw_element("back", 18, 23, mask=self.num_back < 1)

    def add_to_bar(self, item):

        state = 0  # 0 continue, 1 failed to game over, 2 success to game over, 3 has del
        if item in self.bar_items:
            self.bar_items.insert(self.bar_items.index(item), item)
        else:
            self.bar_items.append(item)

        isdel = False
        for item in set(self.bar_items):
            if self.bar_items.count(item) == 3:
                isdel = True
                for i in range(3):
                    del self.bar_items[self.bar_items.index(item)]

        if isdel:
            self.last_value = None
            self.last_operator = None
            self.last_position = None
            self.finished_blocks -= 3
            if self.finished_blocks == 0:
                state = 2
            else:
                state = 3
        elif len(self.bar_items) == self.max_num_bar:
            state = 1
        return state  # 0 continue, 1 failed to game over, 2 success to game over, 3 has del

    def step_serial(self, action):

        assert action >= 0 and action < 334, "Out of action range."
        # action -> [0, 334)
        # body[18x18] + external[3] + stack[4] + button[3]
        idxs = np.cumsum([0, 18 * 18, 3, 4, 3])
        for i in range(len(idxs) - 1):
            lower, upper = idxs[i], idxs[i + 1]
            if action < upper:
                val = action - lower

                # body
                if i == 0:
                    absx = (val % 18) + 1
                    absy = (val // 18) + 1
                elif i == 1:
                    absx = val * 2 + 1
                    absy = 21
                elif i == 2:
                    if self.stack_use_plane:
                        absx, absy = (1 + val * 4), 19
                    else:
                        absx, absy = 1 + val * 4 + (len(self.stack_items[val])-1 if val < len(self.stack_items) else 0) * 0.2, 19
                elif i == 3:
                    # self.num_shuffle, self.num_push, self.num_back
                    absx, absy = [[18, 21], [16, 22], [18, 23]][val]
                break
        return self.step(absx, absy)

    def step_def(self, type, action):

        if type == "body":
            assert action >= 0 and action < 18 * 18, "Out of body range."
            absx = (action % 18) + 1
            absy = (action // 18) + 1
        elif type == "external":
            assert action >= 0 and action < 3, "Out of external range."
            absx = action * 2 + 1
            absy = 21
        elif type == "stack":
            assert action >= 0 and action < 4, "Out of stack range."
            if self.stack_use_plane:
                absx, absy = (1 + action * 4), 19
            else:
                absx, absy = 1 + action * 4 + (
                    len(self.stack_items[action]) - 1 if action < len(self.stack_items) else 0) * 0.2, 19
        elif type == "button":
            assert action >= 0 and action < 3, "Out of button range."
            absx, absy = [[18, 21], [16, 22], [18, 23]][action]
        else:
            raise RuntimeError(f"Unknow type {type}")

        return self.step(absx, absy)

    def step(self, x, y):
        istack = self.hit_stack(x, y)
        body_pos = self.hit_body(x, y)
        button = self.hit_button(x, y)
        iext = self.hit_external(x, y)
        state = -1  # -1 invalid action, 0 continue, 1 failed to game over, 2 success to game over, 3 has del, 4 function button
        if istack != -1:
            item = self.stack_items[istack][-1]
            self.last_operator = "stack"
            self.last_position = istack
            self.last_value = item
            del self.stack_items[istack][-1]
            state = self.add_to_bar(item)
        elif body_pos:
            x, y = body_pos
            ilayer, item, flag = self.layout_squeeze[y, x]

            off = 0
            if ilayer % 2 == 1:
                off = 1
            x = (x - off) // 2 * 2 + off
            y = (y - off) // 2 * 2 + off

            self.last_operator = "body"
            self.last_position = self.layout[y:y + 2, x:x + 2, ilayer, 0]
            self.last_value = self.layout[y:y + 2, x:x + 2, ilayer, 0].copy(), item // 4
            self.layout[y:y + 2, x:x + 2, ilayer, 0] = 0
            del self.blocks[self.blocks.index(item // 4)]
            state = self.add_to_bar(item // 4)
        elif button is not None:
            state = 4
            if button == "shuffle":
                self.shuffle()
                self.num_shuffle -= 1
            elif button == "back":
                if self.last_operator == "body":
                    self.last_position[:] = self.last_value[0]
                    self.blocks.append(self.last_value[1])
                    del self.bar_items[self.bar_items.index(self.last_value[1])]
                    self.num_back -= 1
                elif self.last_operator == "stack":
                    self.stack_items[self.last_position].append(self.last_value)
                    del self.bar_items[self.bar_items.index(self.last_value)]
                    self.num_back -= 1
                elif self.last_operator == "external":
                    self.extrnal_items.insert(self.last_position, self.last_value)
                    del self.bar_items[self.bar_items.index(self.last_value)]
                    self.num_back -= 1
                else:
                    state = -1
            elif button == "push":
                topn = min(3, len(self.bar_items))
                if topn > 0:
                    for i in range(topn):
                        self.extrnal_items.append(self.bar_items[i])

                    for i in range(topn - 1, -1, -1):
                        del self.bar_items[i]
                    self.num_push -= 1
                else:
                    state = -1

            self.last_operator = None
            self.last_value = None
            self.last_position = None
        elif iext is not None:
            item = self.extrnal_items[iext]
            self.last_operator = "external"
            self.last_position = iext
            self.last_value = item
            del self.extrnal_items[iext]
            state = self.add_to_bar(item)
        else:
            state = -1

        self.update()
        rstate = self.get_env_state()
        done = False
        reware = -1

        #  -1 unknow action, 0 continue, 1 failed to game over, 2 success to game over, 3 has del, 4 function button
        if state == -1:
            reware = -0.2
        elif state == 0:
            reware = -1
        elif state == 1:
            reware = -10
            done = True
        elif state == 2:
            reware = 10
            done = True
        elif state == 3:
            reware = 5
        elif state == 4:
            reware = -0.1

        if done:
            self.reset()
        return rstate, reware, done

    def interactive(self):

        def callback(event, x, y, flag, user):
            if event == cv2.EVENT_LBUTTONDOWN:
                (body_elements, body_mask_flag, bar_state, external_state, stack_state, button_state), reware, done = \
                    self.step(x / self.half_size, y / self.half_size)

                print(f"reware={reware}, done={done}, bar_state={bar_state}, button_state={button_state}, external_state={external_state}, stack_state={stack_state}, body_elements={body_elements.shape}, body_mask_flag={body_mask_flag.shape}")
                cv2.imshow("game", self.render())

        cv2.namedWindow("game")
        cv2.setMouseCallback("game", callback)
        self.reset()

        while True:
            cv2.imshow("game", self.render())

            key = cv2.waitKey() & 0xFF
            if key >= ord('1') and key <= ord('9'):
                print(self.layout[:, :, -(key - ord('0')), 0])
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()