from .base_options import BaseOptions


class SoccerOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--video', type=str, help='input video file path')
        self.parser.add_argument('--home_club_id', type=str, help='club id of the home team')
        self.parser.add_argument('--away_club_id', type=str, help='club id of the away team')
        self.isTrain = False
