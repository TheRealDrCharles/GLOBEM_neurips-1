def _init_ema_weights(self):
    for param in self.get_ema_model().parameters():
        param.detach_()
    mp = list(self.get_model().parameters())
    mcp = list(self.get_ema_model().parameters())
    for i in range(0, len(mp)):
        if not mcp[i].data.shape:  # scalar tensor
            mcp[i].data = mp[i].data.clone()
        else:
            mcp[i].data[:] = mp[i].data[:].clone()

def _update_ema(self, iter):
    alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
    for ema_param, param in zip(self.get_ema_model().parameters(),
                                self.get_model().parameters()):
        if not param.data.shape:  # scalar tensor
            ema_param.data = \
                alpha_teacher * ema_param.data + \
                (1 - alpha_teacher) * param.data
        else:
            ema_param.data[:] = \
                alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]
