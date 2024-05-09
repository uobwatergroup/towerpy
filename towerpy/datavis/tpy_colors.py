"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

"""
This file contains a dictionary listing different colour gradients.
"""

towerpy_colours = {
    'rad_ref': ['#967BB6', '#A690C4', '#B6A6D1', '#C6BBDF', '#D6D1EC',
                # Lavender Gradient Color (R)
                '#51D8FF', '#46C7FF', '#3DB5FF', '#239BFE', '#1672F9',
                '#044EE3',  # Russian Blues Color Scheme
                '#FFF480', '#FFDF00', '#F6D108', '#EDC211', '#E3B419',
                '#DAA521',  # Golden Yellow Gradient Color Scheme
                '#FF9533', '#FF7F34', '#FE6D35', '#FF5836', '#FF4137',
                '#B31B1B',  # NAME: SoundCloud Colors
                '#5E0400', '#F8E8EC'],
    'rad_pvars': ['#D2ECFA', '#51D8FF', '#46C7FF', '#3DB5FF', '#239BFE',
                  '#1672F9', '#044EE3',  # Russian Blues Color Scheme
                  '#FFFF00', '#FFF480', '#FFE961', '#FFDF41', '#FFD422',
                  '#FFC903',  # Vivid Yellow Gradient Color Scheme
                  '#FFB829', '#F98A2C', '#ED5B45', '#B64262', '#633663',
                  '#402852',  # Towerpy
                  '#C6BFCB'],
    'rad_rainrt': ['#D2ECFA', '#51D8FF', '#46C7FF', '#3DB5FF', '#239BFE',
                   '#1672F9', '#044EE3',  # Russian Blues Color Scheme
                   '#44AEB3', '#249EA0', '#008B8B', '#007979', '#006767',
                   # Dark Cyan Monochromatic
                   '#FFFF00', '#FFF480', '#FFE961', '#FFDF41', '#FFD422',
                   '#FFC903',  # Vivid Yellow Gradient Color Scheme
                   '#FFB829', '#F98A2C', '#ED5B45', '#B64262', '#633663',
                   '#402852',  # Towerpy
                   '#C6BFCB'],
    'rad_pvars_g': ['#D2ECFA', '#51D8FF', '#46C7FF', '#3DB5FF', '#239BFE',
                    '#1672F9', '#044EE3',  # Russian Blues Color Scheme
                    '#1A724C', '#A9FC02',  # Spring Greens
                    '#FFFF00', '#FFF480', '#FFE961', '#FFDF41', '#FFD422',
                    '#FFC903',  # Vivid Yellow Gradient Color Scheme
                    '#FFB829', '#F98A2C', '#ED5B45', '#B64262', '#633663',
                    '#402852',  # Towerpy
                    '#C6BFCB'],
    'rad_2slope': ['#505050', '#666a6c', '#7b8489', '#919ea5', '#b1c5d0',
                   '#bcd2de', '#d2ecfa', '#46C7FF', '#3DB5FF', '#239BFE',
                   '#1672F9', '#044EE3',  # Russian Blues Color Scheme
                   '#FFF480', '#FFE961', '#FFDF41', '#FFD422', '#FFC903',
                   # Vivid Yellow Gradient Color Scheme
                   '#FFB829', '#F98A2C', '#ED5B45', '#B64262', '#633663',
                   '#402852',  # Towerpy
                   '#C6BFCB'],
    'div_yw_gy_bu': ['#509AE8', '#505050', '#FFBA01'],
    'div_dbu_rd': ['#DA2C43', '#E15566', '#E97E88', '#F0A8AB', '#F8D1CD',
                   '#AFC6D9', '#83A3BE', '#5880A2', '#2C5D87', '#003A6B'],
    'div_lbu_w_rd': ['#DA2C43', '#E15566', '#E97E88', '#F0A8AB', '#F8D1CD',
                     '#FFFAF0', '#F0FFFE', '#C0E5FE', '#90CCFE', '#60B2FE',
                     '#3099FE', '#007FFE'],
    'div_dbu_w_rd': ['#DA2C43', '#E15566', '#E97E88', '#F0A8AB', '#F8D1CD',
                     '#FFFAF0', '#F0FFFE', '#AFC6D9', '#83A3BE', '#5880A2',
                     '#2C5D87', '#003A6B'],
    'div_dbu_w_k': ['#2B2B2B', '#545454', '#808080', '#AFAFAF', '#E1E1E1',
                    '#FCFFFA', '#F0FFFE', '#AFC6D9', '#83A3BE', '#5880A2',
                    '#2C5D87', '#003A6B'],
    'div_rd_w_k': ['#2B2B2B', '#545454', '#808080', '#AFAFAF', '#E1E1E1',
                   '#FCFFFA', '#FFFAF0', '#F8D1CD', '#F0A8AB', '#E97E88',
                   '#E15566', '#DA2C43'],
    'useq_grey': ['#505050', '#5b5d5e', '#666a6c', '#71777b', '#7b8489',
                  '#869197', '#919ea5', '#9cabb3', '#a7b8c1', '#b1c5d0',
                  '#bcd2de', '#c7dfec', '#d2ecfa'],
    'useq_ywbu': ['#FFBA01', '#d0a819', '#99872B', '#666D41', '#335456',
                  '#003A6B'],
    'useq_bupkyw': ['#3F3A81', '#954698', '#D65A94', '#F76385', '#F9B357',
                    '#F8F658'],
    'useq_morning': ['#F6BD73', '#F9D69E', '#CBB6B0', '#9DA3B7', '#5373A1',
                     '#3D5688'],
    'useq_wblk': ['#F2F3F4', '#B3BDD4', '#7587B4', '#365194', '#253354',
                  '#141414'],
    'useq_fiery': ['#FCD988', '#F9A622', '#F55E01', '#EB2701', '#C40000',
                   '#292929'],
    'useq_tec': ['#B0F7FF', '#9BCFE3', '#86A7C7', '#7080AA', '#5B588E',
                 '#463072'],
    'useq_pastel': ['#FCF5E3', '#FCD4B8', '#F3AE9C', '#D799A7', '#9A7DA7',
                    '#4E60A4'],
    'useq_wk': ['#F8F8FF', '#CAC9CD', '#9B9A9C', '#6D6A6A', '#3E3B39',
                '#100C07'],
    'useq_sun': ['#FFF673', '#FFDB02', '#FFBC11', '#FF9D1F', '#FF7E2E',
                 '#FF6352'],
    'useq_calm': ['#F6EEAB', '#C9DD94', '#9DCF94', '#7EC796', '#5EBD96',
                  '#11A797'],
    'useq_model': ['#ffffff', '#efefef', '#dfdfdf', '#cfcfcf', '#bfbfbf',
                   '#b0b0b0', '#a1a1a1', '#929292', '#838383', '#757575',

                   '#00305a', '#113f6b', '#204f7d', '#2d608e', '#3a71a0',
                   '#4783b3', '#5595c5', '#62a8d8', '#70bbea', '#7ecefd',

                   '#ffee58', '#ffe148', '#ffd339', '#ffc52a', '#ffb71b',
                   '#ffa80c', '#ff9900', '#ff8900', '#ff7800', '#ff6600',

                   '#ff1d23', '#ec1820', '#d9141d', '#c60f1a', '#b30b17',
                   '#a10714', '#8f0411', '#7e020d', '#6d0108', '#5c0002',

                   '#3a0f57', '#4a1e62', '#592c6d', '#683a78', '#784984',
                   '#87588f', '#96689b', '#a677a8', '#b587b4', '#c498c1']
    }
