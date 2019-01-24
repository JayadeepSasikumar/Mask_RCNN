require(dplyr)
require(ggplot2)

# Bar chart plotting the different attributes and the number of video
# sequences annotated with each attribute
attrs <- c('DEF', 'LR', 'SV', 'SC', 'FM', 'CS', 'IO', 'MB', 'OCC', 'HO',
                 'EA', 'OV', 'BC', 'DB', 'AC')
cardinality <- c(30, 13, 16, 21, 21, 13, 25, 17, 18, 37, 23, 10, 9, 12, 11)
davis.attributes <- data.frame(attrs, cardinality)
qplot(x=attrs, y=cardinality, data=davis.attributes,
      group=attrs, geom='blank',
      main = 'Attributes vs no. of sequences - DAVIS 2016 dataset') +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Attributes") + ylab('Number of video sequences') +
  theme_classic()


# Training split - attributes and video sequences
train.attrs <- c('DEF', 'LR', 'SV', 'SC', 'FM', 'CS', 'IO', 'MB', 'OCC',
                 'HO', 'EA', 'OV', 'DB', 'BC', 'AC')
train.cardinality <- c(17, 8, 8, 14, 14, 6, 15, 8, 12, 23, 12, 6, 9, 5, 4)
train.split <- data.frame(train.attrs, train.cardinality)
qplot(x=train.attrs, y=train.cardinality, data=train.split,
      group=train.attrs, geom='blank',
      main = 'Attributes vs no. of sequences - Training split') +
  ylim(0, 25) +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  xlab("Attributes") + ylab('Number of video sequences') +
  theme_classic()

# Validation split - attributes and video sequences
val.attrs <- c('LR', 'SV', 'SC', 'FM', 'CS', 'IO', 'MB', 'DEF', 'OCC', 'HO',
               'EA', 'BC', 'DB', 'OV', 'AC')
val.cardinality <- c(5, 8, 7, 7, 7, 10, 9, 13, 6, 14, 11, 4, 3, 4, 7)
val.split <- data.frame(val.attrs, val.cardinality)
qplot(x=val.attrs, y=val.cardinality, data=val.split,
      group=val.attrs, geom='blank',
      main = 'Attributes vs no. of sequences - Validation split') +
  ylim(0, 25) +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Attributes") + ylab('Number of video sequences') +
  theme_classic()

# Test split - attributes and video sequences
test.attrs <- c('SV', 'SC', 'FM', 'IO', 'MB', 'DEF', 'HO', 'EA', 'OV', 'AC',
               'LR', 'OCC')
test.cardinality <- c(4, 2, 2, 4, 2, 4, 5, 3, 1, 3, 1, 1)
test.split <- data.frame(test.attrs, test.cardinality)
qplot(x=test.attrs, y=test.cardinality, data=test.split,
      group=test.attrs, geom='blank',
      main = 'Attributes vs no. of sequences - Test split') +
  ylim(0, 25) +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Attributes") + ylab('Number of video sequences') +
  theme_classic()


# Validation subset - post split - attributes and video sequences
split.val.attrs <- c('LR', 'SV', 'SC', 'FM', 'CS', 'IO', 'MB', 'DEF', 'OCC', 'HO',
                     'EA', 'BC', 'DB', 'OV', 'AC')
split.val.attrs.cardinality <- c(4, 4, 5, 5, 7, 6, 7, 9, 5, 9, 8, 4, 3, 3, 4)
split.val.attrs.split <- data.frame(split.val.attrs, split.val.attrs.cardinality)
qplot(x=split.val.attrs, y=split.val.attrs.cardinality, data=split.val.attrs.split,
      group=split.val.attrs, geom='blank',
      main = 'Attributes vs no. of sequences - Validation split') +
  ylim(0, 25) +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Attributes") + ylab('Number of video sequences') +
  theme_classic()

# Mean IOUs over sequences
video.sequences <- c('bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn',
                     'dance-jump', 'dog-agility', 'drift-turn', 'elephant', 'flamingo',
                     'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly',
                     'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding', 'rhino',
                     'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing',
                     'tennis', 'train', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
                     'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
                     'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump',
                     'paragliding-launch', 'parkour', 'scooter-black', 'soapbox')
dataset <- c('train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train',
             'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train',
             'train', 'train', 'train', 'train', 'train', 'validation', 'validation', 'validation',
             'validation', 'validation', 'test', 'test', 'test', 'test', 'test', 'test', 'test',
             'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test',
             'test', 'test')
elu.ious <- c(0.9136543299768529, 0.13143337034576774, 0.7356186661881311, 0.8163531978547891,
          0.8531447086150322, 0.9363159507816796, 0.7659785797501365, 0.8143406103873352,
          0.9066999171261847, 0.8997630326774317, 0.7740653418332168, 0.8758600074416331,
          0.7534878954519451, 0.785007808122164, 0.6274281553026724, 0.8396621891680018,
          0.8187484636712904, 0.9063559215224665, 0.7936458643557004, 0.807310624376892,
          0.8516273617837877, 0.8804976682454904, 0.810807759238688, 0.8091475505087833,
          0.8961169215548299, 0.3826509202933891, 0.8648072991986907, 0.7707944566842919,
          0.7816693387628769, 0.7539771907304429, 0.7460476689613976, 0.4731706034212429,
          0.43107716977439914, 0.5790563858334041, 0.9395416109133924, 0.9162215939927427,
          0.8559921598348518, 0.7198102271104724, 0.8551628165796834, 0.8546575762631221,
          0.8961718608675339, 0.8079687082477803, 0.7515617289083097, 0.6004798870795697,
          0.7360749606905908, 0.6233498646016721, 0.6064516011134164, 0.8522469214898356,
          0.8092873030492972, 0.48463671029560884)
sequence.wise.ious <- data.frame(video.sequences, dataset, ious)
qplot(x=video.sequences, y=ious, data=filter(sequence.wise.ious, dataset == 'test'),
      group=video.sequences, geom='blank',
      main = 'Mean Jaccard index over each video sequence in test dataset - ELU') +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Video sequence") + ylab('Mean Jaccard index') +
  theme_classic() +
  theme(axis.text.x = element_text(angle=60, hjust = 1, vjust = 1, size = 13))


# Final training split
attrs <- c('DEF', 'LR', 'SV', 'SC', 'FM', 'CS', 'IO', 'MB', 'OCC', 'HO',
           'EA', 'OV', 'DB', 'BC', 'AC')
cardinality <- c(14, 8, 6, 11, 10, 4, 11, 7, 11, 18, 11, 5, 8, 5, 4)
davis.attributes <- data.frame(attrs, cardinality)
qplot(x=attrs, y=cardinality, data=davis.attributes,
      group=attrs, geom='blank',
      main = 'Attributes vs no. of sequences - Final training split') +
  ylim(0, 25) +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Attributes") + ylab('Number of video sequences') +
  theme_classic()

# Final validation split
attrs <- c('SC', 'FM', 'CS', 'IO', 'DEF', 'HO', 'SV', 'DB', 'OV', 'OCC',
           'MB', 'EA')
cardinality <- c(3, 4, 2, 4, 3, 5, 2, 1, 1, 1, 1, 1)
davis.attributes <- data.frame(attrs, cardinality)
qplot(x=attrs, y=cardinality, data=davis.attributes,
      group=attrs, geom='blank',
      main = 'Attributes vs no. of sequences - Final validation split') +
  ylim(0, 25) +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Attributes") + ylab('Number of video sequences') +
  theme_classic()

# Testing set results - ReLU
sequences <- c('bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus',
               'car-turn', 'dance-jump', 'dog-agility', 'drift-turn',
               'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low',
               'kite-walk', 'lucia', 'mallard-fly', 'mallard-water',
               'motocross-bumps', 'motorbike', 'paragliding', 'rhino',
               'rollerblade', 'scooter-gray', 'soccerball', 'stroller',
               'surf', 'swing', 'tennis', 'train', 'blackswan', 'bmx-trees',
               'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
               'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat',
               'horsejump-high', 'kite-surf', 'libby', 'motocross-jump',
               'paragliding-launch', 'parkour', 'scooter-black', 'soapbox')
dataset <- c('train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train',
             'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train',
             'train', 'train', 'train', 'train', 'train', 'validation', 'validation', 'validation',
             'validation', 'validation', 'test', 'test', 'test', 'test', 'test', 'test', 'test',
             'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test',
             'test', 'test')
ious <- c(0.9136543299768529,
          0.13143337034576774,
          0.7356186661881311,
          0.8163531978547891,
          0.8531447086150322,
          0.9363159507816796,
          0.7659785797501365,
          0.8143406103873352,
          0.9066999171261847,
          0.8997630326774317,
          0.7740653418332168,
          0.8758600074416331,
          0.7534878954519451,
          0.785007808122164,
          0.6274281553026724,
          0.8396621891680018,
          0.8187484636712904,
          0.9063559215224665,
          0.7936458643557004,
          0.807310624376892,
          0.8516273617837877,
          0.8804976682454904,
          0.810807759238688,
          0.8091475505087833,
          0.8961169215548299,
          0.3826509202933891,
          0.8648072991986907,
          0.7707944566842919,
          0.7816693387628769,
          0.7539771907304429,
          0.7478864133433651,
          0.47540883664381356,
          0.44093785975977073,
          0.5871835075920948,
          0.9403326696942552,
          0.916593820511908,
          0.8556838396715559,
          0.7203357845335431,
          0.8545649303112716,
          0.854459453370028,
          0.8974399425609035,
          0.8080275982882419,
          0.7516226119380668,
          0.5993657458572655,
          0.7366984290345949,
          0.6230072304821879,
          0.6068223818415123,
          0.85288931345508,
          0.810252355442376,
          0.49254346828795026)
sequence.wise.ious <- data.frame(sequences, dataset, ious)
qplot(x=sequences, y=ious, data=filter(sequence.wise.ious, dataset == 'test'),
      group=sequences, geom='blank',
      main = 'Mean Jaccard index over each video sequence in test dataset - ReLU') +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Video sequence") + ylab('Mean Jaccard index') +
  theme_classic() +
  theme(axis.text.x = element_text(angle=60, hjust = 1, vjust = 1, size = 13))


# bmx-tree IoUs
ious <- c(0.6160776047454077,
         0.6304177731817378,
         0.6309205193444667,
         0.6165756837722197,
         0.5942094482965683,
         0.6563169938931815,
         0.5913558304116349,
         0.6140983495985252,
         0.6182661435028939,
         0.6255736555623729,
         0.6145168118943326,
         0.6208280862677176,
         0.6247291987944564,
         0.6213634616531587,
         0.578249708522776,
         0.39434286420371106,
         0.29828918049022035,
         0.026272007473960662,
         0.44915895242446696,
         0.5413756813338111,
         0.6255888844811978,
         0.5140149809421187,
         0.39563830741748757,
         0.48350848956450776,
         0.46798385384831875,
         0.5665834782917392,
         0.5788005358557505,
         0.552988604921008,
         0.5551467142403529,
         0.5341692045451734,
         0.5850444689608041,
         0.5336905030121477,
         0.5138905323452283,
         0.5840535812251273,
         0.5110986400298791,
         0.4680490341502624,
         0.4448029006199938,
         0.49161624532781795,
         0.5825706035048945,
         0.5895041075403329,
         0.6041484757487054,
         0.6652904479141271,
         0.6579626426808852,
         0.6063959119187559,
         0.6107935578825825,
         0.6410039125592835,
         0.661621267229381,
         0.6302891523548471,
         0.6060110905322776,
         0.5781410143388134,
         0.6180014940382419,
         0.39810176269093367,
         0.0,
         0.0,
         0.36257975447409774,
         0.4109246131564906,
         0.42262160974455476,
         0.0,
         0.0,
         0.3634684136187182,
         0.35341184600912245,
         0.0,
         0.0,
         0.0,
         0.0,
         0.43685371521171906,
         0.4899152549456507,
         0.39159496505988256,
         0.0,
         0.0,
         0.5706109069147393,
         0.6375682637665262,
         0.6298308578122582,
         0.6492570324471041,
         0.5983686418116595,
         0.6577693498354611,
         0.6827517016006495,
         0.6886420848112885,
         0.6876454361433676,
         0.6423584529114239)
frames <- 1:80
bmx <- data.frame(frames, ious)
qplot(x=frames, y=ious, data=bmx,
      group=frames, geom='blank',
      main = 'Jaccard index over each frame in bmx-tree video sequence') +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Frame") + ylab('Jaccard index') +
  theme_classic() +
  theme(axis.text.x = element_text(angle=60, hjust = 1, vjust = 1, size = 13))


# Test sequences - length and ious
test.sequences <- c('blackswan', 'bmx-trees', 'breakdance', 'camel',
                    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl',
                    'dog', 'drift-chicane', 'drift-straight', 'goat',
                    'horsejump-high', 'kite-surf', 'libby', 'motocross-jump',
                    'paragliding-launch', 'parkour', 'scooter-black', 'soapbox')
test.lengths <- c(50, 80, 84, 90, 75, 40, 104, 90, 60, 52,
                  50, 90, 50, 50, 49, 40, 80, 100, 43, 99)
test.ious <- c(0.7478864133433651, 0.47540883664381356, 0.44093785975977073,
               0.5871835075920948, 0.9403326696942552, 0.916593820511908,
               0.8556838396715559, 0.7203357845335431, 0.8545649303112716,
               0.854459453370028, 0.8974399425609035, 0.8080275982882419,
               0.7516226119380668, 0.5993657458572655, 0.7366984290345949,
               0.6230072304821879, 0.6068223818415123, 0.85288931345508,
               0.810252355442376, 0.49254346828795026)
length.df <- data.frame(seq=test.sequences, len=test.lengths, iou=test.ious)
qplot(x=len, y=iou, data=length.df, geom='point',
      main = 'Sequence length vs mean Jaccard index') +
  geom_point(alpha = 0.4, colour='#0A51FF') +
  # geom_text(aes(label = cardinality)) +
  xlab("Sequence length") + ylab('Jaccard index') +
  theme_classic()


# Attribute-wise IoU
v.attrs <- c('AC', 'BC', 'CS', 'DB', 'DEF', 'EA', 'FM', 'HO',
             'IO', 'LR', 'MB', 'OCC', 'OV', 'SC', 'SV')
attr.ious <- c(0.7015529503772254, 0.8469775538083449, 0.6792770140976638,
               0.7638757013555729, 0.7227167390494779, 0.7393131452159163,
               0.6755021431872069, 0.703860432521824, 0.6697532622270904,
               0.6643711231724787, 0.6070666234125656, 0.7395314271757609,
               0.6540991100446042, 0.6627396053583624, 0.689669020021249)
attr.df <- data.frame(v_attr=v.attrs, iou=attr.ious)
qplot(x=v_attr, y=iou, data=attr.df,
      group=v_attr, geom='blank',
      main = 'Mean Jaccard index over different attributes') +
  geom_bar(stat = 'identity', position = 'dodge', alpha = 0.4, fill='#0A51FF') +
  xlab("Attribute") + ylab('Jaccard index') +
  theme_classic() +
  theme(axis.text.x = element_text(angle=60, hjust = 1, vjust = 1, size = 13))
