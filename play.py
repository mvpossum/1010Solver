# Starting game loop
############################################################################
import human_input, game, random_strat, greedy_alg, naive_alg, time, datetime
import curses
solver_dict = {"human":human_input, "random":random_strat, "greedy":greedy_alg, "naive":naive_alg, "replay":None}
#~ solver = raw_input("Please select a solver from the following by typing its name\n{}\n".format(solver_dict.keys()))
solver = 'greedy'


stdscr = curses.initscr()
curses.start_color()
curses.noecho()
curses.cbreak()
#curses.curs_set(0)
stdscr.keypad(1)
curses.mousemask(curses.ALL_MOUSE_EVENTS)
begin_x = 0; begin_y = 0
height = 30; width = 60
win = curses.newwin(height, width, begin_y, begin_x)



#~ if solver == "replay":
	#~ score = game.replay(raw_input("enter replay string:\n"))
if solver in solver_dict:
    while True:
        start = time.clock()
        score = game.play(win, solver_dict[solver].get_move)
        end = time.clock()
        elapsed = end - start
	# game.Result(score[2], start, elapsed, score[3], [solver])
#~ print("Game Over! You score {} points, lasted {} moves and cleared {} lines!\nThe game lasted {}".format(score[2],score[0],score[1], str(datetime.timedelta(seconds=elapsed))))

curses.nocbreak(); stdscr.keypad(0); curses.echo()
curses.endwin()
